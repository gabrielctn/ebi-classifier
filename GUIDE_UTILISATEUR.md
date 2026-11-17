# EBI Classifier — user guide / guide utilisateur

## Windows

If you received the Windows ZIP, extract **all** of it to a writable folder such as Documents. Open the extracted folder and double-click **START_EBI.bat**. Keep the black window open while the application runs. First-time setup can take several minutes; the browser should open automatically. If it does not, use the local address printed in the window. Close the black window to stop the application.

The Windows package targets Windows 10 or 11, 64-bit Intel/AMD PCs. It includes Python, the model, and the required packages and normally works without an internet connection. Allow approximately 3 GB of free space. Some computers may need the Microsoft Visual C++ Runtime installed by IT. See `START_HERE.txt` in the package for full instructions and troubleshooting in English and French.

### Classifying images

1. Check **Channel Mapping** matches your experiment. Defaults are erythroblasts = 7, macrophages = 11 and neutrophils = 3. Select three different channel numbers.
2. Upload the three separate TIFF images for each sample. Their sample ID must match, for example `sample001_Ch7.ome.tif`, `sample001_Ch11.ome.tif` and `sample001_Ch3.ome.tif`.
3. Click **Run Classification**. Compare the processed count in the summary with the number of samples you submitted: incomplete triplets are skipped.
4. Open **View Prediction Details** to inspect and, if needed, move images between the EBI and NOT_EBI groups.
5. Download **EBI IDs (.pop)** after reviewing. It contains the IDs currently in the EBI group, one per line; it is empty when no samples are in that group.

The application runs locally on the computer and uses a local browser address. Predictions are a research aid: review the results and assess suitability for your own experiment.

## macOS

If you received the Mac ZIP, double-click it to extract the folder, open that folder, and double-click **START_EBI.command**. Keep the Terminal window open while you use the browser interface described above. The first start can take several minutes. Close Terminal to stop the application.

The Mac package supports Intel and Apple Silicon Macs running macOS 13 or later. It includes Python, the model, and the required packages; normal setup and use work offline. Allow approximately 3 GB of free space. See `START_HERE.txt` in the package for English and French instructions, including what to do if macOS blocks the app.

## Linux

Extract the entire Linux ZIP, open the extracted folder, and double-click **EBI_Classifier.desktop**. If the desktop asks, allow it to launch. Keep the terminal window open while using the browser interface, and close the terminal to stop the app. If the desktop launcher does not run, open a terminal in the extracted folder and run `./START_EBI.sh`. See `START_HERE.txt` for detailed instructions.

The Linux package supports 64-bit Intel/AMD and ARM computers running a glibc-based Linux distribution with glibc 2.28 or later. It includes Python, the model, and dependencies for offline use. Allow approximately 4 GB of free space.

## Français

Sur Windows, faites un clic droit sur le ZIP, choisissez **Extraire tout**, ouvrez le dossier extrait puis double-cliquez sur **START_EBI.bat**. Gardez la fenêtre noire ouverte. Le premier démarrage peut prendre plusieurs minutes. Le navigateur s'ouvre automatiquement ; sinon, ouvrez l'adresse locale indiquée dans la fenêtre. Fermez la fenêtre noire pour quitter. `START_HERE.txt` contient les instructions et le dépannage en français.

Vérifiez les trois canaux dans **Channel Mapping** : par défaut Ch7 pour les érythroblastes, Ch11 pour les macrophages et Ch3 pour les neutrophiles. Déposez les trois TIFF de chaque échantillon, puis cliquez sur **Run Classification**. Vérifiez le nombre d'échantillons traités : les triplets incomplets sont ignorés. Examinez les résultats dans **View Prediction Details** puis téléchargez le fichier `.pop` des identifiants EBI après vérification.

Sur Mac, décompressez le ZIP pour Mac, ouvrez le dossier extrait et double-cliquez sur **START_EBI.command**. Gardez Terminal ouvert pendant l'utilisation et fermez-le pour quitter. Le fichier `START_HERE.txt` contient les instructions et le dépannage en français.

Sur Linux, extrayez tout le ZIP et double-cliquez sur **EBI_Classifier.desktop** dans le dossier extrait. Autorisez son lancement si nécessaire. Sinon, ouvrez un terminal dans le dossier et saisissez `./START_EBI.sh`. Gardez le terminal ouvert et fermez-le pour quitter. Voir `START_HERE.txt` pour les détails.
