# Wstępna implementacja III projektu

## Cel projektu

Celem tego projektu jest wytrenowanie modeli sztucznej inteligencji do przewidywania pogody.

## Dane

Dane (które można znaleźć na przykład [tutaj](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data)) należy umieścić w folderze ./data/
Dane dotyczą pogody w kilkudziesięciu miastach, głównie w Stanach Zjednoczonych.

## Oznaczenie miast

Miasta w tym projekcie będziemy czasami oznaczali trzema pierwszymi literami nazwy miasta,
wyjątkiem są miasta zaczynające się od słowa "San", które są oznaczane pierwszymi sześcioma literami.

## Predykcje na podstawie sąsiadów

W tym projekcie każde miasto ma własną, wytrenowaną tylko dla siebie sieć neuronową.
Jako dane wejściowe takiej sieci neuronowej są warunki pogodowe z trzech poprzednich dni nie tylko w
mieście, w którym pogodę chcemy przewidywać, ale również w kilku miastach sąsiadujących.
To, jakie miasta traktowane są jako sąsiedzi danego miasta zapisane jest w pliku ./models/cities.json. Jako pierwszy sąsiad danego miasta zawsze wymienione jest najpierw to samo miasto.

## Wytrenowane modele

W folderze ./models/, oprócz opisanego wcześniej pliku cities.json, znajdują się wytrenowane modele
dla poszczególnych miast (z wyjątkiem Vancouver).  Dla każdego miasta znajduje się tam plik modelu do predykcji temperatury oraz jeden plik modelu do predykcji siły wiatru. Dodatkowo do każdego modelu znajduje się plik "normalizer", który przechowuje informacje o preprocessingu (normalizacji) danych do sieci, a także informacje o architekturze danej sieci.

## Jak korzystać?

Załóżmy, że chcemy zobaczyć predykcje dla miasta Indianapolis. Na początku wejdźmy do pliku ./models/cities.json aby zobaczyć, na podstawie jakich miast tworzone są predykcje. Skrót dla miasta Indianapolis to "Ind" (pierwszy w pliku), zatem jego sąsiedzi to Indianapolis, Saint Louis, Kansas City, Chicago, Pittsburgh, Philadelphia. Na początku musimy przygotować dane treningowe. W tym celu wchodzimy do pliku ./data_proc.py i w linijce 29 wpisujemy odpowiednie miasta. Uruchamiamy plik ./data_proc.py, na mapie widzimy wybrane miasta, a nastepnie uruchamiamy ./window_slasher.py. Teraz w folderze clean_norm_data/concat_clean_data_simulate_middle_day_test powinny znajdować się dane treningowe oraz testowe dotyczące miasta Indianapolis. Następnie w pliku ./torch_test.py w linijce 8 zmieniamy "city" na trzyliterowy skrót Indianapolis czyli "Ind", a w linijce 9 wpisujemy "temp" lub "wind" w zależności od tego, co chcemy przewidywać. Następnie uruchamiamy plik ./torch_test.py.

## Trenowanie modeli

Do trenowania modeli służy plik ./torch_train.py. Należy w nim wybrać trzyliterowy skórt miasta, kolumnę jaką chce się przewidywać ("temp" lub "wind") a także liczbę epok. Program autoamtycznie zapisze model, a także jego normalizer w folderze ./models/ pod odpowiednią nazwą.

