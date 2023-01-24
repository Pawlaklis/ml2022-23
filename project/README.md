# Temat: 2048 napisane w Pythonie zgodne z Gymnasium
## Paweł Knapczyk
Gra napisana w Pythonie, stworzone środowisko, na którym zostaną przetestowane przykładowe algorytmy do RL

### Cel:
Model podpięty pod grę ma "wygrać" grę (połączyć klocki do otrzymania 2048) 

### Dane:
Algorytmy Stable Baselines (3-5?) - https://github.com/hill-a/stable-baselines  
Gymnasium API - https://github.com/Farama-Foundation/Gymnasium

DDPG

# Wykonanie projektu
### Implementacja środowiska
W związku z tym że Stable Baselines3 nie działa jeszcze z Gymnasium byłem zmuszony użyć gym'a

Folder game2048 zawiera klasę naszego środowska które implementuje zasady gry. Nasza gra posiada 2 systemy wyświetlania, gdy chcemy pokazać graczowi plansze, oraz gdy przeprowadzamy uczneie

Zbudowane środowisko musimy zarejestrować i zainstalować jako dependencję

Folder logs zawiera logi uczenia które możemy przeglądać przy użyciu Tensorboard 

Folder models zawiera zapisane które wytrenowaliśmy

Proces uczenia opisany jest w project.ipynb