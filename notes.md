# TODOS

## Fragen

## Last time
- Mit Bartzi gechatte

## Ansätze finden
- [X] Code/Models für GAN anfragen
- [X] GAN-Paper nochmal lesen und verstehen und dazu nochmal schreiben
- [X] Ref 15 anschauen
- [] GAN bauen
  - [] Netzwerkgraphen malen (vor allem für Generator)
  - [] in pytorch examples reinschauen

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

