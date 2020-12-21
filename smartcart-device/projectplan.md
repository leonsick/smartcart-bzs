#### Project plan SmartCart device

- Objekterkennung die konstant Objekte erkennt
- Waage die konstant das Gewicht im Wagen misst
    - Die Waage muss Veränderung im Gewicht erkennen
    
Eingesetzt als: <br>
Erkennung von neuen Item:
- Wenn Objekt erkannt:
    - Warte auf Gewichtszunahme
    - Wenn Gewichtszunahme geschehen:
        - Prüfe Gewichtszunahme im Vgl. zu Gewicht pro Stück -> Füge Anzahl hinzu
        
Erkennung der Entnahme eines Items:
- Wenn Gewichtsreduktion erkannt:
    - Warte auf Objekterkennung:
        - Wenn Objekt mit Confidence > 0.7 erkannt:
            - Entferne Objekte entsprechend der Gewichtsabnahme vom Wagen
            

Speicherung in Datenbank:
- Erstellung einer DynamoDB
- Speichert ab:
    - 1 Tabelle pro Cart mit Daten:
        - ItemId
        - Einheit (ob kg oder Stück)
        - Anzahl
        - Gewicht
        - Priced by Weight
        - Timestamp
    - Löschung des Inhalts der Tabelle nach Checkout

Speicherung des Sortiments:
- Erstellung von Item Objekten
- Diese Speichern:
    - Name
    - Preis
    - Gewicht pro Stück
    - Bepreist nach Gewicht (Boolean)
   
   
##TFLite Runtime Links:
###Raspberry Python 3.7
https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl

###Mac Python 3.7
https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37-macosx_10_15_x86_64.whl

###Mac Python 3.8
https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp38-cp38-macosx_10_15_x86_64.whl

        
