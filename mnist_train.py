import torch                                                                    # carica PyTorch
from torchvision import datasets, transforms                                    # dataset pronti e conversione delle immagini
from torch.utils.data import DataLoader, random_split                           # consegna a batch e divisione dei dati
from torch import nn                                                            # pezzi per costruire le reti

device = "mps" if torch.backends.mps.is_available() else "cpu"                  # usa la GPU del Mac se c'è
print(device)                                                                   # stampa dove avvengono i calcoli
torch.manual_seed(42)                                                           # fissa pesi iniziali e mescolata del DataLoader

train_full = datasets.MNIST(
    "data",                                                                     # cartella dei file
    train=True,                                                                 # le 60.000 immagini di training
    download=True,                                                              # scarica i dati solo se mancano
    transform=transforms.ToTensor(),                                            # converte ogni immagine in tensor con valori da 0 a 1
)
print(len(train_full))                                                          # quante immagini ci sono

image, label = train_full[0]                                                    # prende la prima immagine e la sua cifra
print(image.shape)                                                              # forma: 1 canale, 28 righe, 28 colonne
print(label)                                                                    # la cifra della prima immagine

generator = torch.Generator().manual_seed(42)                                   # con lo stesso seed esce sempre la stessa divisione
train_set, val_set = random_split(                                              # divide a caso il dataset
    train_full,                                                                 # il dataset da dividere
    [50000, 10000],                                                             # 50.000 training, 10.000 validation
    generator=generator,                                                        # usa il seed fissato sopra
)
print(len(train_set))                                                           # immagini di training
print(len(val_set))                                                             # immagini di validation

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)               # consegna 64 immagini alla volta, mescolate
val_loader = DataLoader(val_set, batch_size=256)                                # consegna la validation a gruppi di 256
print(len(train_loader))                                                        # quanti batch ci sono: 782

images, labels = next(iter(train_loader))                                       # prende il primo batch: immagini e risposte giuste
print(images.shape)                                                             # 64 immagini da [1, 28, 28]
print(labels.shape)                                                             # 64 risposte giuste

model = nn.Sequential(                                                          # mette gli strati in fila
    nn.Flatten(),                                                               # da griglia 28×28 a lista di 784 numeri
    nn.Linear(784, 128),                                                        # da 784 numeri a 128: qui ci sono pesi da imparare
    nn.ReLU(),                                                                  # azzera i negativi: permette di combinare i pattern
    nn.Linear(128, 10),                                                         # da 128 numeri a 10 punteggi, uno per cifra
).to(device)                                                                    # sposta la rete sulla GPU
print(model)                                                                    # stampa la struttura della rete

images = images.to(device)                                                      # sposta il batch sulla GPU
outputs = model(images)                                                         # calcola 10 punteggi per ognuna delle 64 immagini
print(outputs.shape)                                                            # forma: 64 immagini, 10 punteggi
print(outputs[0])                                                               # i 10 punteggi della prima immagine del batch
print(labels[0])                                                                # la cifra giusta della prima immagine del batch

loss_fn = nn.CrossEntropyLoss()                                                 # crea la funzione che misura l'errore
labels = labels.to(device)                                                      # sposta le risposte giuste sulla GPU
loss = loss_fn(outputs, labels)                                                 # errore medio sulle 64 immagini
print(loss.item())                                                              # stampa la loss: circa 2,30 con pesi casuali

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)                      # crea il correttore dei pesi
optimizer.zero_grad()                                                           # azzera i gradient vecchi
loss.backward()                                                                 # calcola il gradient di ogni peso
optimizer.step()                                                                # corregge tutti i pesi di un piccolo passo

new_output = model(images)                                                      # rifà la previsione con i pesi nuovi
new_loss = loss_fn(new_output, labels)                                          # misura di nuovo l'errore
print(new_loss.item())                                                          # dovrebbe essere più basso di prima

def evaluate(loader):                                                           # definisce una funzione: misura loss e accuracy su un loader
    model.eval()                                                                # mette la rete in modalità valutazione
    total_loss = 0                                                              # somma delle loss di tutte le immagini
    correct = 0                                                                 # contatore delle risposte giuste
    total = 0                                                                   # contatore delle immagini viste
    with torch.no_grad():                                                       # niente gradient: qui non si impara
        for images, labels in loader:                                           # per ogni batch del loader ricevuto
            images, labels = images.to(device), labels.to(device)               # sposta il batch sulla GPU
            outputs = model(images)                                             # 10 punteggi per ogni immagine
            total_loss += loss_fn(outputs, labels).item() * len(labels)         # loss media del batch × immagini = loss totale del batch
            correct += (outputs.argmax(dim=1) == labels).sum().item()           # aggiunge le risposte giuste del batch
            total += len(labels)                                                # aggiunge le immagini del batch
    return total_loss / total, correct / total                                  # restituisce loss media e accuracy


best_val_loss = float("inf")                                                    # miglior val loss vista finora: parte da infinito
patience = 3                                                                    # quante epoch senza miglioramenti prima di fermarsi
epochs_without_improvement = 0                                                  # contatore delle epoch senza miglioramenti

for epoch in range(1, 21):                                                      # al massimo 20 epoch: epoch vale da 1 a 20
    model.train()                                                               # mette la rete in modalità addestramento
    for images, labels in train_loader:                                         # per ognuno dei 782 batch di training
        images, labels = images.to(device), labels.to(device)                   # sposta il batch sulla GPU
        optimizer.zero_grad()                                                   # azzera i gradient del batch precedente
        loss = loss_fn(model(images), labels)                                   # previsione e calcolo dell'errore
        loss.backward()                                                         # calcola il gradient di ogni peso
        optimizer.step()                                                        # corregge i pesi
    train_loss, train_acc = evaluate(train_loader)                              # misura sul training, a epoch finita
    val_loss, val_acc = evaluate(val_loader)                                    # misura sulla validation, a epoch finita
    print("epoch", epoch)                                                       # numero dell'epoch
    print("  train: loss", round(train_loss, 4), "acc", round(train_acc, 4))    # risultati sul training, 4 decimali
    print("  val:   loss", round(val_loss, 4), "acc", round(val_acc, 4))        # risultati sulla validation, 4 decimali

    if val_loss < best_val_loss:                                                # la val loss è la più bassa finora?
        best_val_loss = val_loss                                                # sì: diventa il nuovo record
        epochs_without_improvement = 0                                          # azzera il contatore
        torch.save(model.state_dict(), "best_model.pt")                         # salva i pesi in un file
        print("  salvato: nuovo miglior modello")                               # avvisa che ha salvato
    else:                                                                       # no: la rete non è migliorata
        epochs_without_improvement += 1                                         # aumenta il contatore di 1
        if epochs_without_improvement == patience:                              # 3 epoch di fila senza miglioramenti?
            print("stop: nessun miglioramento da", patience, "epoch")           # avvisa che si ferma
            break                                                               # esce dal ciclo delle epoch


test_set = datasets.MNIST("data", train=False, download=True,transform=transforms.ToTensor())                    # le 10.000 immagini di test, mai usate finora convertite in tensor come le altre
test_loader = DataLoader(test_set, batch_size=256)                              # consegna il test a gruppi di 256

model.load_state_dict(torch.load("best_model.pt"))                              # rimette nella rete i pesi dell'epoch migliore
test_loss, test_acc = evaluate(test_loader)                                     # misura sul test, una volta sola
print("test: loss", round(test_loss, 4), "acc", round(test_acc, 4))             # risultato finale