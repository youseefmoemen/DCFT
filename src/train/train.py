from src.model.dcft import DCFT
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchinfo import summary


def load_data(tokenizer, batch_size=32):
    train_data = load_dataset('nyu-mll/glue', 'cola', split='train')
    validation_data = load_dataset('nyu-mll/glue', 'cola', split='validation')
    tokenized_train = train_data.map(lambda example: tokenizer(example['sentence'], padding=True, truncation=True), batched=True, batch_size=batch_size, remove_columns='sentence')
    tokenized_validation = validation_data.map(lambda example: tokenizer(example['sentence'], padding=True, truncation=True), batched=True, batch_size=batch_size, remove_columns='sentence')
    tokenized_train.set_format('torch')
    tokenized_validation.set_format('torch')
    train_dataloader = DataLoader(tokenized_train, batch_size=batch_size)
    validation_dataloader = DataLoader(tokenized_validation, batch_size=batch_size)
    return train_dataloader, validation_dataloader



def load_base_model(model_name, label_map):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=label_map['id2label'], label2id=label_map['label2id'])
    return tokenizer, model

def get_dcft_mode(model, d, k):    
    
    for layer in model.deberta.encoder.layer:
        attn = layer.attention.self
        for attr in ["in_proj", "pos_proj", "pos_q_proj"]:
            setattr(attn, attr, DCFT(layer=getattr(attn, attr), d=d, k=k))
    
    for name, param in model.named_parameters():
        if not any(module in name for module in ["in_proj", "pos_proj", "pos_q_proj"]) or 'base_layer' in name:
            param.requires_grad = False
    return model



def main():
    print('Loading DCFT Model')
    label_map = {'label2id': {'acceptable': 1, 'unacceptable': 0}, 'id2label': {0: 'unacceptable', 1: 'acceptable'}}
    model_name = 'microsoft/deberta-base'
    d, k = 8, 1
    tokenizer, base_model = load_base_model(model_name=model_name, label_map=label_map)
    dcft_model = get_dcft_mode(base_model, d, k)
    print('Loading Cola Data')
    train_dataloader, validation_dataloader = load_data(tokenizer)
    #    print('Number of trainable params', dcft_model.print_trainable_parameters())
    batch = next(iter(train_dataloader))
    batch.pop('label')
    batch.pop('idx')
    print(dcft_model(**batch))
    print(summary(dcft_model, input_data=dict(batch)))
    return dcft_model



# model.deberta.encoder.layer[2].attention.self.in_proj

model = main()