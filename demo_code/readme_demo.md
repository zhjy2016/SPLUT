## Dependency

- Python 3
- Android Studio

## Code

- `MainActivity.java`: the main activity code for the demos
- `acitivity_main.xml`: the code for the interface layout
- `gen_csv.py`: the code for transfering `.npy` LUT files to `.csv` LUT files

## Generating APK

1. Run `gen_csv.py` to create the LUT files for demos

2. Create a new project named `SR-LUT`/`SPLUT-S`/`SPLUT-M`/`SPLUT-L` in Android Studio and enter the root folder of the project. 

3. Create a folder of `./app/src/main/assets` and place the `.csv` files, the `Set14_HR` folder with Set14 HR images and the `Set14_LR` folder with Set14 LR images in `./app/src/main/assets`

4. Replace `./app/src/main/java/com/example/.../MainActivity.java` by the `MainActivity.java` file provided by us. 

5. Replace `./app/src/main/res/layout/activity_main.xml` by the `activity_main.xml` file provided by us. 

6. Add the following code in `./app/src/main/AndroidManifest.xml`:

   ```
   <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE">
   </uses-permission>
   ```

7. Set the Sdk version number to no more than 28 in `./app/build.gradle`
8. Generate Signed APK and evaluate the demos on mobile phones

