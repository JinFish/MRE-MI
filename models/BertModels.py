from transformers import BertModel, ViTModel
import torch
import math
    
class RA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.attention_head_size = int (self.hidden_size / self.num_attention_heads)

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        
        self.text_projection = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.image_projection = torch.nn.Linear(self.hidden_size, self.hidden_size)
        
        self.w_1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.w_2 = torch.nn.Linear(self.hidden_size, 1)
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    # (bsz, seq_len, hidden_size) -> (bsz, num_heads, seq_len, head_size)
    def transpose_for_scores(self, x:torch.Tensor):
        # (bsz, seq_len, hidden_size) -> (bsz, seq_len, num_heads, head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(*new_x_shape)
        # (bsz, seq_len, num_heads, head_size) -> (bsz, num_heads, seq_len, head_size)
        return x.permute(0,2,1,3)
    
    def forward(self, text, image):
        bsz = text.shape[0]
        text_len = text.shape[1]
        image_len = image.shape[1]
        # (bsz, text_len, dim)
        text_emb = self.text_projection(text)
        # (bsz, img_len, dim)
        image_emb = self.image_projection(image)
        # (bsz, text_len, dim) -> (bsz, text_len, 1, dim)
        text_emb = text_emb.unsqueeze(2)
        # (bsz, img_len, dim) -> (bsz, 1, img_len, dim)
        image_emb = image_emb.unsqueeze(1)
        # (bsz, text_len, img_len, dim) 
        relavance = torch.mul(text_emb, image_emb)
        # (bsz, text_len, img_len, dim) 
        relavance = self.relu(self.w_1(relavance))
        # (bsz, 1, text_len ,img_len)
        relavance = self.sigmoid(self.w_2(relavance)).reshape(bsz, 1, text_len, image_len)
        
        mixed_query_layer = self.query(text)
        mixed_key_layer = self.key(image)
        mixed_value_layer = self.value(image)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # (bsz, num_heads, seq_len1, head_size) * (bsz, num_heads, head_size, seq_len2)
        # -> (bsz, num_heads, seq_len1, seq_len2)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        

        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = relavance * attention_probs
        # (bsz, num_heads, seq_len1, seq_len2) * (bsz, num_heads, seq_len2, head_size)
        # (bsz, num_heads, seq_len1, head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (bsz, num_heads, seq_len1, head_size) -> (bsz, seq_len1, num_heads, head_size)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        # (bsz, seq_len1, num_heads, head_size) -> (bsz, seq_len1, hidden_size)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer


# Global-Local Relevance-Modulated Attention Network (GLRAMNet)
class GLRA(torch.nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super().__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.text_model_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.vit = ViTModel.from_pretrained(args.image_model_name)
        self.vit.eval()
        self.vit_config = self.vit.config
        self.num_labels = num_labels
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(self.vit_config.hidden_size, 768)
        self.ra = RA()
        self.fc = torch.nn.Linear(768*6, self.num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<h>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<t>")
        self.tokenizer = tokenizer

        

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, main_images=None, aux_images=None, labels=None):
        # (bsz, 4, 3, 224, 224) + (bsz, 12, 3, 224, 224) -> (bsz, 16, 3, 224, 224)
        pixel_values = torch.cat([main_images, aux_images], dim=1)
        bsz, img_len, channels, height, width = pixel_values.shape
        # (bsz, 16, 3, 224, 224) -> (bsz*16, 3, 224, 224)
        pixel_values = pixel_values.reshape(bsz*img_len, channels, height, width)
        with torch.no_grad():
            img_features = self.vit(pixel_values=pixel_values)[1]
        # (bsz*16, 768) -> (bsz, 16, 768)
        img_features = img_features.reshape(bsz, img_len, -1)
        img_features = self.fc1(img_features)
        # (bsz, 16, 768) -> (bsz, 4, 768)
        global_image_representation = img_features[:,0:4,:]
        # (bsz, 16, 768) -> (bsz, 12, 768)
        local_image_representation = img_features[:, 4:,:]

        
        
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)  
        # (bsz, seq) -> (bsz, seq) -> [row(0-bsz), column(0-seq)]
        head_indexs = torch.eq(input_ids, self.head_start).nonzero(as_tuple=True)
        # bsz
        head_rows = head_indexs[0]
        # bsz
        head_columns = head_indexs[1]
        tail_indexs = torch.eq(input_ids, self.tail_start).nonzero(as_tuple=True)
        # bsz
        tail_rows = tail_indexs[0]
        # bsz
        tail_columns = tail_indexs[1]

        sequence_output = bert_output[0]  # bsz, len, hidden
        # sequence_output = self.dropout(sequence_output)
        # (bsz, 1, hidden_size)
        global_text_representation = sequence_output[:,0:1,:]
        # (bsz, seq, hidden) -> (bsz, hidden)
        head_entity_output = sequence_output[head_rows,head_columns,:]
        tail_entity_output = sequence_output[tail_rows,tail_columns,:]
        # (bsz, hidden) + (bsz, hidden) -> (bsz, 2, hidden)
        local_text_representation = torch.stack([head_entity_output, tail_entity_output], dim=1)
        
        # (bsz, 1, hidden) 
        global_multimodal_representation = self.ra(global_text_representation, global_image_representation)
        # (bsz, 2, hidden)
        local_multimodal_representation = self.ra(local_text_representation, local_image_representation)
        # (bsz, 1, hidden) + (bsz, 2, hidden), + (bsz, 1, hidden), + (bsz, 2, hidden) -> (bsz, 6*hidden)
        final_representation = torch.cat([global_text_representation,
                                          local_text_representation,
                                          global_multimodal_representation,
                                          local_multimodal_representation], dim=1).reshape(bsz, -1)
        

        logits = self.fc(final_representation)
        loss = None
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels.view(-1))
        
        return loss, logits, labels
    