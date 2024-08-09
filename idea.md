weights retina und neuronen -4 und 4

- neuronen index out nodes bug

- retina nicht auf ranod werte initilaisieren, eher fast alle 0, 10 % nicht

wie können nodes in and out in die netzlisten calculation rein?
linien erkennung und über die position der netzwerke?
örtliche Nähe?

retina größe anpassen? 5x5 , 7x7

- gewichte über die Zeit anzeigen lassen! Mittwelwert und standarabweichung + uzeitschritt über alle Gewichte.

1. Retina runterkslaieren -> super pixel 3x3 parametrisieren
2. node bug fixen
3. mehr Bilder, training und testen

otsu binarization!

50 update schritte oder mehr
___
Idee
Agent hat 2 netzwerke, eins steuert die bewegung , das andere erkennt bauteile

fitness des ersten: wie viel prozent des netzwerkes wurde abgedeckt.

erstmal nur steuern evolvieren

Daten alle x y pos preparieren. Wenn retina nahe dran ist, dann abhakt
Bilder machen mit Liste von allen x y positionen.


https://www.kaggle.com/datasets/johannesbayer/cghd1152

