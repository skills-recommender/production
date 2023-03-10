import json
import re
import logging
import numpy as np
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

class Main:
    def __init__(self):
        self.MAX_LEN = 500
        #self.EPOCHS = 6
        #self.NUM_LABELS = 12
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_PATH = './models/bert-base-uncased'
        self.STATE_DICT = torch.load('./models/model_e6.tar', map_location=self.DEVICE)
        #tokenizer = BertTokenizerFast('../input/bert-base-uncased/vocab.txt', lowercase=True)
        self.tokenizer = BertTokenizerFast('./models/bert-base-uncased/vocab.txt',lowercase=True)
        self.model = BertForTokenClassification.from_pretrained(MODEL_PATH,state_dict=self.STATE_DICT['model_state_dict'], num_labels=12)
        self.model.to(self.DEVICE);

        self.tags_vals = ["UNKNOWN", "O", "Name", "Degree","Skills","College Name","Email Address","Designation","Companies worked at","Graduation Year","Years of Experience","Location"]
        #self.tag2idx = {t: i for i, t in enumerate(tags_vals)}
        self.idx2tag = {i:t for i, t in enumerate(self.tags_vals)}
        
        #self.data = self.trim_entity_spans(self.convert_dataturks_to_spacy('./data/annotated_resumes.json'))

        #self.model = MODEL
        

    def process_resume(self,text):
        max_len = self.MAX_LEN
        tok = self.tokenizer.encode_plus(text, max_length=max_len, return_offsets_mapping=True)
        
        curr_sent = dict()
        
        padding_length = max_len - len(tok['input_ids'])
            
        curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
        curr_sent['token_type_ids'] = tok['token_type_ids'] + ([0] * padding_length)
        curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)
        
        final_data = {
            'input_ids': torch.tensor(curr_sent['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(curr_sent['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(curr_sent['attention_mask'], dtype=torch.long),
            'offset_mapping': tok['offset_mapping']
        }
        
        return final_data

    def predict(self,test_resume):
        return self._predict(self.model,test_resume)
        
    def _predict(self,model,test_resume):
        model.eval()
        data = self.process_resume(test_resume)
        input_ids, input_mask = data['input_ids'].unsqueeze(0), data['attention_mask'].unsqueeze(0)
        labels = torch.tensor([1] * input_ids.size(0), dtype=torch.long).unsqueeze(0)
 
        with torch.no_grad():
            outputs = model(
                input_ids,
                token_type_ids=None,
                attention_mask=input_mask,
                labels=labels,
            )
            tmp_eval_loss, logits = outputs[:2]
        
        logits = logits.cpu().detach().numpy()
        label_ids = np.argmax(logits, axis=2)
        
        entities = []
        for label_id, offset in zip(label_ids[0], data['offset_mapping']):
            curr_id = self.idx2tag[label_id]
            curr_start = offset[0]
            curr_end = offset[1]
            if curr_id != 'O':
                if len(entities) > 0 and entities[-1]['entity'] == curr_id and curr_start - entities[-1]['end'] in [0, 1]:
                    entities[-1]['end'] = curr_end
                else:
                    entities.append({'entity': curr_id, 'start': curr_start, 'end':curr_end})
        for ent in entities:
            ent['text'] = test_resume[ent['start']:ent['end']]
        return entities

        
