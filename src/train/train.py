from src.model.dcft import DCFT
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchinfo import summary
from tqdm.auto import tqdm
import torch

def load_data(tokenizer, batch_size=32):
    train_data = load_dataset('nyu-mll/glue', 'cola', split='train')
    validation_data = load_dataset('nyu-mll/glue', 'cola', split='validation')
    tokenized_train = train_data.map(lambda example: tokenizer(example['sentence'], padding=True, truncation=True), batched=True, batch_size=batch_size, remove_columns=['sentence', 'idx'])
    tokenized_validation = validation_data.map(lambda example: tokenizer(example['sentence'], padding=True, truncation=True), batched=True, batch_size=batch_size, remove_columns=['sentence', 'idx'])
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
        for attr in ["in_proj", "pos_proj"]:
            setattr(attn, attr, DCFT(layer=getattr(attn, attr), d=d, k=k))
    
    for name, param in model.named_parameters():
        if not any(module in name for module in ["in_proj", "pos_proj"]) or 'base_layer' in name:
            param.requires_grad = False
    return model

def train(model, num_epochs, optimizer, scheduler, train_dataloader, validation_dataloader, criterion, device="cuda", max_grad_norm=1.0):

    model.to(device)

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        total_loss = 0.0

        for batch in loop:
            labels = batch.pop('label').to(device)  
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            
            logits = model(**batch).logits
            loss = criterion(logits, labels)

            loss.backward()
            
            clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            global_step += 1

            if global_step % 100 == 0:
                val_loss, val_acc = evaluate(model, validation_dataloader, criterion, device)
                print(f"\nStep {global_step}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} finished | Avg train loss: {avg_loss:.4f}")


def evaluate(model, dataloader, criterion, device="cuda"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop('label').to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch).logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    model.train()  # switch back to training mode
    return avg_loss, accuracy

def main():
    print('Loading DCFT Model')
    label_map = {'label2id': {'acceptable': 1, 'unacceptable': 0}, 'id2label': {0: 'unacceptable', 1: 'acceptable'}}
    model_name = 'microsoft/deberta-base'
    d, k = 8, 1
    EPOCHS = 2
    lr = 1e-4
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer, base_model = load_base_model(model_name=model_name, label_map=label_map)
    dcft_model = get_dcft_mode(base_model, d, k)
    
    print('Loading Cola Data')
    train_dataloader, validation_dataloader = load_data(tokenizer, batch_size=batch_size)
    batch = next(iter(train_dataloader))
    batch.pop('label')
    print(summary(dcft_model, input_data=dict(batch)))


    optimizer = AdamW(dcft_model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dataloader)*EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    dcft_model = train(
        model=dcft_model,
        num_epochs=EPOCHS,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        criterion=criterion, 
        device=device
    )
    return dcft_model



if __name__=="__main__":
    model = main()
    