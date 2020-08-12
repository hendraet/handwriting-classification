# TODOS

- [X] Mischen Handschrift printed text
- [X] labels plotten
- [X] Alphanum:
  - Großbuchstaben, Unterstriche, Zahlen 
  - erst random dann mit Regelmäßigkeit
- [X] visual backprop (kein linear layer zwischendrin)
- [ ] Monatswörter

## Fragen

**07.08.**
- hard classifier?
- soft wie? Hannes Ansatz nur binär
- Hannes dist to score?
- ganz anderer Ansatz? (log reg trainieren?)

**06.05.**
- Wie rausfinden, welcher Teil Dinge kaputt macht? - erst mal runterbrechen auf simplere Bild-Generierung (K=1/3-channel-Input)
- Würde anderer Discriminator nen Unterschied machen? - ja, Discriminator sind relativ speziell und funktionieren nur mit den Gen zusammen
- strided conv in dis? - kein Gamechanger
--> Update Prozess anschauen, evtl. mal die Standard-GAN-Impl von Pytorch anschauen

**26.04.**
Issues/solutions vanishing gradient:
- tanh/Sigmoid functions (check activation functions)
- batch norm helps (more batchnorm?)
- Gradient noise addition?
- do I train ResBlocks correctly (read somewhere that gradient should add up)
- learning rate changes?
- weight initialization
  - Xavier for tanh
  - kaiming for convs (at least in res_blocks)
  - constant for weight (1) and bias (0)

**24.04.**
- der gaze optimizer-Kram
- zu viele dims von Wörtern (640)? eher weniger?

--> gradients, weigths plotten (tensorboard)
--> 320
--> Testweise: lr stark runtersetzen, batch_size = 1
--> Bildbreit/größe runternehmen (Bartzi: 200 x 64) - Nöpe (wörter overflowen sonst)

**old**
- Wie stell ich die conv-Parameter im Discriminator ein?
- Wo soll das avg pooling im Discriminator hin?
  - Einfach einen ans Ende
  - Any intuition for values? - Einfach schauen was andere gemacht
- ResBlocks:
  - Feature Map halbiert sich nach jeder Ebene, Channels verdoppeln sich
- Ist der Classifier einfach ein flacher fully connected layer? - ja
- Plan:
  - conv
  - 6 mal \_makenet aber mit 4er ResBlocks + avg pooling -
  - Fc

- Wie funktioniert die Loss calc für das seq2seq Ding?
  - Netzwerk wir nur auf echten Daten optimiert, aber nicht pretained. Gleichzeitig werden auch irgendwie die generierten Daten reingesteckt. Wie passt das zusammen?
  - Loss-Calc-Formel sieht so aus, als würde das sowohl auf echten, als auch auf generierten Daten passieren. (x~{X^-,X})
  - Ist Optim. des Netzwerks ein seperater Prozess und wirdd dafür einfach ein anderer Loss genutzt?
  - **batch-wise, erst alle Discriminator optimieren und dann generative Network**

## Last time
- Nur ein paar kleinere Tweaks am Generative Network, wie Upsampling, Hidden Layer für MLP researched, normailiserung der Bilder auf -1;1
- mit Bartzi weitere Dinge besprochen

%%%%%%%%%%%%%%%%%%%%%

## seq2seq analysis
- new dataset from genreated images - merged with train dataset of real images

## GANwriting
- [X] Embedding size auf 128 runternehmen
- [X] expand und repeat verifien
- [X] VGG19 zum Laufen bekommen
- [X] cwe-output channels runternehmen und chars zusammenstacken
- [X] mal höhere Bilder für VGG-Input testen - 672x96 mit 5er upsampling läuft
- [X] Embedding kleiner machen so auf 32
- [] Gaussian-Noise fixen
- [] irgendwas zwischen 350 und 672 width reinwerfen und schauen
- [] crop the word-iamdb-dataset down to 60k samples
- [] figure out how empty char is encoded in case of loss comp

## Ansätze finden
- [X] Code/Models für GAN anfragen
- [X] GAN-Paper nochmal lesen und verstehen und dazu nochmal schreiben
- [X] Ref 15 anschauen
- [X] GAN bauen
  - [X] Netzwerkgraphen malen (vor allem für Generator)
  - [X] in pytorch examples reinschauen

## Datensatzgenerierung
- [X] Fontsize (oder image size) dynamisch anpassen
- [X] Daten, Währung und Shit random generieren
- [X] Fonts runterladen/von Bartzi holen
- [X] Input Dimensions
- [X] wordlist nehmen für text generation
- [X] Groundtruth in JSON speichern
- [X] Datensatz generieren (Bilder auf Platte, Groundtruth in JSON), 10k pro Klasse
- [X] verschiedene Datumsformate generieren
- [X] Kurze (einstellige) Zahlen generieren
- [X] Handschrift (Evaluierungs-)Datensatz generiern z.B. mit dem iam(hist)db
  - [X] fix padding in data set
  - [X] upload to server
  - [X] get evaluation running (dataset generation)
- [X] den echten IamDB Datensatz anschauen
- [X] Bilder als Grayscale und nicht als RGB speichern (braucht wahrscheinlich Änderungen im Trainingscode)
- [X] Data Aug auf iamdb für mehr Bilder
  - [X] bei rotation die schwarzen Pixel ersetzen/mirroren
  - [X] doch piecewise (mit mode edge)
  - [X] shearing bisschen erhöhen
  - [X] auch words brightness technisch anpassen
- [X] offline Handwriting von rrnlib anschauen
- [X] Punkte, Zahlen und Unterstriche für Synthese-Netzwerk extrahieren
- [X] "Zahlwörter" generieren, die das Format 19.12.2020 sein (eng zusammengeschrieben!) haben sollten
  - Preproc kann sein, dass es keine kompletten Datumsangaben rausgibt sondern nur Zahlen/einzelne Blöcke, ganze Datumsangaben wären tortzdem nett
- [X] einfach mal orig Netwerk mit zusammengesetzten Dates trainieren -> klappt nicht, lässt sich zu gut unterscheiden
  - [X] Strokes zu Bildern der korrekten Größe konvertieren

- [?] Fix cropping/spacing so that words completely fit in image
- [?] "Verhandschriftlichung" der Daten (mal nach Offline handwriting, iam hat auch online)
- [?] mal schauen, wie man noch anders Handwriting generieren kann, wenn online nicht funktioniert
- [?] auch mal online handwriting von anderen Sprachen anschauen

- [] für Training Sätze in einzelne Wörter zerhacken

## Netzwerke
- [X] Hannes Datensatz auf Chainer MNIST umbiegen und zum Laufen bekommen
- [X] PCA drauf werfen und schauen, ob Cluster entstehen
- [X] mal schauen, ob das auch schon mit meinen Bildern funktioniert - negativ, brauch vermutlich besseres Netz
- [X] auch auf Server zum Laufen bekommen
- [X] Resnet 18 drauf werfen
- [X] Zahlen vs Buchstaben unterscheiden
- [X] Datum vs (Zahlen, Buchstaben)
- [X] Netzwerk auf Iamdb daten generieren
- [X] Loss plotten (ext)
- [X] Cluster über epochen plotten (ext schreiben)
- [X] Bilder in Cluster Plot statt Punkte
- [X] **REFACTOR**

- [] Guten Wert für den margin-Hyperparameter finden
- [] andere Netzwerke?
    - AlexNet, VGG, InceptionNet?
    - ResNet Size zum Test mal erhöhen

## Plotter
- [] draw function analog zu clustering.py refactorn

## Evaluierung
- [] bei einigen Punkten (die z.B. weiter als th von centroids sind) sagen, dass man sich bei diesen unsicher ist -> evtl auch confidence angeben
- [] challenge finden auf der man evaluieren kann

## Sonstiges
- [X] Hannes MA lesen
- [X] Related Work zu Date Field Extraction lesen
- [X] Related Work Doktorarbeit Mandal lesen
- [X] schauen, ob es irgendne cluster related work zu triplet/siamese networks gibt

- [X] **Be awesome**

-------------------------------------

#Genereller Apporach:
- Klassifizierung von erkannter Handschrift in Archiv-Dokumenten, z.B. sehen wir hier gerade ein Datum, handelt es sich um eine Unterschrift
- in Zusammenarbeit mit dem Wildenstein Plattner Institut, welche uns Scans echter Archivdokumente zur Verfügung stellen
- Problem: Wir haben keinen großen Datensatz mit annotierten Dokumenten

#Lösungsansatz:
- Bereits verfügbare Methoden zur Generierung von Handschrift nutzen, um daraus einen Datensatz zu generieren
- darauf ein neuronales Netzwerk genrieren und dieses dann nutzen, um Handschrift auf den originalen Daten/Bildern zu klassifizieren

Idee: Embedinngs trainieren - z.B. Datumsangaben nah beiander
- dafür könnte man auch erstmal verschieden gedruckte Fonts verwenden und darauf was zu trainieren
- Stichwort: triplet-loss (anchor (echtes Bild) und dann ein positives + negatives und daraus dann den Loss berechnen)
Related Work anschauen

# Klassen
- Numerisch
  - Preis
  - Datum
  - Seitenzahl
  - Zipcode
- alphanumerisch (erstmal als eine Klasse und nicht die Unterklassen an sich betrachten)
  - Dossier Nummern (keine schematische Eingrenzung möglich)
  - Hausnummern
- alphabetisch
  - Unterschriften
  - Initials
- anderes
  -Skizzen, Zeichnungen

# Sonstige Infos
- github link für Hannes shit: https://github.com/hrantzsch/signature-verification
- GPU unter 172.20.8.50 (overview) und 172.20.8.14 (ssh)

# Vortrag (20 min)
- Motivation (wie bei Meinel)
- Related Work vorstellen
  - Unterscheidung da die meisten gelabelte Daten haben
  - vor allem auf Hannes eingehen
- Übersicht über Analyse-Pipeline zeigen, da ich ja davon ausgehe, gecroppte Wörter zu bekommen
- Experimente zeigen
- Outlook (generierung von Handschrift, entweder hier oder bei related work)

# Infos
- nvidia-smi um Grafikarten anzuzeigen

# Fragen
- Handschrift nach Autoren durchsuchen oder nach Autoren unterscheiden (kann das Netzwerk das vlt automatisch)
- aufpassen, dass unterschiedliche Handschrift generiert wird

