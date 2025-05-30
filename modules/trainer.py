import torch
import time
from torch import optim
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup 
from sklearn.metrics import classification_report as sk_classification_report
from .metrics import eval_result

class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()


class RETrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, args=None,
                 logger=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.logger = logger
        self.re_dict = processor.get_relation_dict()
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_dev_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args


    def train(self):
        self.before_train()

        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))

                true_labels, pred_labels = [], []

                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    loss, logits, labels = self._step(batch)
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()


                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())


                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)

                        avg_loss = 0

                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values())[1:],
                                                     target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("------------  This is the Training Result ------------")
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                self.logger.info("The Training acc = %s, The Training f1 = %s" % (str(acc), str(micro_f1)))


                self.evaluate(epoch)  # generator to dev.

            pbar.close()
            self.pbar = None

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    loss, logits, labels = self._step(batch)
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values())[1:],
                                                     target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1  # update best metric(f1 score)
                    torch.save(self.model.state_dict(), f"best_model.pth")

        self.model.train()

    def test(self):
        self.model.to(self.args.device)
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        
        self.model.load_state_dict(torch.load(f"best_model.pth"))
        self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    loss, logits, labels = self._step(batch)  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())

                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values())[1:],
                                                     target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)

                total_loss = 0
                self.logger.info("Test f1 score: {}, acc: {}.".format(micro_f1, acc))

        self.model.train()

    def _step(self, batch):
        input_ids, token_type_ids, attention_mask, main_images, aux_images, labels = batch
        loss, logits, labels = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            main_images=main_images, aux_images=aux_images, labels=labels) 

        return loss, logits, labels
        

    def before_train(self):
        parameters = []
        params = {'lr': self.args.lr, 'weight_decay':1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if "vit" not in name:
                params['params'].append(param)
        parameters.append(params)
        self.optimizer = optim.AdamW(parameters, lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)