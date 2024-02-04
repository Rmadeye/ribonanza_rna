# kaggle_rna
zbieramy pieniadze na pront do EDI
### uwagi
* nowy model nowy plik w katalogu `network/models` nie ruszamy działających modeli, na któych były już uczone modele
* nowe dane są tutaj `/home/nfs/kkaminski/kaggle/rna_ready` zawiera `reactivity_err.pt  reactivity_mask.pt  reactivity.pt  sequences.pt` zapisane jako tensory nie trzeba ich już modyfikować
# * czy jesteśmy pewni, że padding powinien być  wypełniany zerami, skoro wg słownika A=0  ?
## TODO
### opublikowanie pierwszych wyników
 * [X] przerobienie `baseline.ipynb` na skrypt, który będzie przyjmował 4 parametry `hparams.yml`, folder z danymi, folder do zapisu danych z uczenia oraz flagę `--test` która ma na celu ułatwienie debugowania kodu. Po odpaleniu dodaniu flagi, model ma być uczony 1 epokę na małym procencie danych tak aby szybko sprwadzić czy wszystko działa
 * [X] dodać zapis modelu na końcu skryptu, zapisywany ma być `state_dict` oraz hparams ostatniej epoki
 * [X] stworzyć klasę która wczytuje model ze `state_dict` wraz z metodą predict, która otrzemyje `DataLoader` i zwraca predykcje w postaci `[num_seq, 206]`
 * [X] opublikować wyniki na kaggle za pomocą ich cli
 ### poprawa modelu
 * [X] podział na train/test na razie losowy najlepiej aby random seed podziału był ustalony i znajdował się w `hparams.yml`
 * [X] dodanie `lr_scheduler` np. tego https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
 * [X] przerywanie uczenia jeśli model przestał się uczyć od `N` epok
 * [X] dodać funkcjonalność która zapisuje do katalogu wynikowego model z najlepszym `val_loss/test_loss` a nie ostatni
 * [X] dodanie argparse
 * [ ] usunięcie padowania przy liczeniu loss
 
