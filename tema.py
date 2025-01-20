import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Setarea dispozitivului (GPU sau CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Încărcarea datelor din fișierele CSV
train_df = pd.read_csv('train.csv', encoding='utf-8')
test_df = pd.read_csv('test.csv', encoding='utf-8')

# Funcție de preprocesare pentru text
def preprocess(text):
    return text.lower()

# Aplicarea funcției de preprocesare asupra textelor din seturile de antrenament și test
train_df['Text'] = train_df['Text'].apply(preprocess)
test_df['Text'] = test_df['Text'].apply(preprocess)

# Maparea etichetelor la valori numerice
label_dict = {'true': 0, 'fake': 1, 'biased': 2}
train_df['Label_num'] = train_df['Label'].map(label_dict)

# Inițializarea tokenizer-ului și modelului BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3).to(device)

# Funcție pentru tokenizare
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Împărțirea datelor în seturi de antrenament și validare
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['Text'], train_df['Label_num'], test_size=0.2, random_state=42, stratify=train_df['Label_num']
)

# Tokenizarea textelor
train_encodings = tokenize_function(train_texts.tolist())
val_encodings = tokenize_function(val_texts.tolist())

# Crearea DataLoader-elor pentru antrenament și validare
train_dataset = TensorDataset(
    train_encodings['input_ids'].to(device),
    train_encodings['attention_mask'].to(device),
    torch.tensor(train_labels.tolist()).to(device)
)
val_dataset = TensorDataset(
    val_encodings['input_ids'].to(device),
    val_encodings['attention_mask'].to(device),
    torch.tensor(val_labels.tolist()).to(device)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Configurarea optimizer-ului și a funcției de pierdere
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Antrenarea modelului
num_epochs = 5
best_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
    
    # Evaluare pe setul de validare
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(val_true, val_preds)
    print(f"\nEpoch {epoch+1}/{num_epochs} - Accuracy: {accuracy}")
    print(classification_report(val_true, val_preds, target_names=['true', 'fake', 'biased']))

    # Salvarea modelului cu cea mai bună acuratețe
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model_state.bin')
    
    scheduler.step()

# Încărcarea modelului cu cea mai bună acuratețe
model.load_state_dict(torch.load('best_model_state.bin'))

# Predictii pe datele de test
test_encodings = tokenize_function(test_df['Text'].tolist())
test_dataset = TensorDataset(
    test_encodings['input_ids'].to(device),
    test_encodings['attention_mask'].to(device)
)
test_loader = DataLoader(test_dataset, batch_size=16)

model.eval()
test_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids, attention_mask = batch
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        test_preds.extend(preds.cpu().numpy())

# Convertirea predictiilor înapoi în etichete text
label_dict_reverse = {v: k for k, v in label_dict.items()}
test_df['Label'] = [label_dict_reverse[pred] for pred in test_preds]

# Salvarea predictiilor în fișierul test.csv original
test_df.to_csv('test_completed.csv', index=False)

print("\nPredictiile au fost salvate în fișierul 'test_completed.csv'")
