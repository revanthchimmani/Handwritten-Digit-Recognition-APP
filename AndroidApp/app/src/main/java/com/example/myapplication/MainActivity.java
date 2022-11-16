package com.example.myapplication;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    private TextView pagenameTextView;
    private Button nextButton, openCameraButton;
    private ImageView imageView;
    String selectedImagePath;
    Bitmap selectedImageBitmap;
    Uri selectedImageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        pagenameTextView = findViewById(R.id.pagename);
        nextButton = findViewById(R.id.next);
        openCameraButton = findViewById(R.id.cameraButton);
        imageView = findViewById(R.id.imageView);


        openCameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openCamera(view);
                openCameraButton.setText("Retake Image");
            }
        });

        nextButton.setEnabled(false);

        nextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                nextPage(view);
            }
        });

        OkHttpClient okHttpClient = new OkHttpClient();

        Request request = new Request.Builder().url(getString(R.string.ip_address)).build();

        okHttpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this, "server_down", Toast.LENGTH_SHORT).show();
                        pagenameTextView.setText("Error connecting to the server");
                    }
                });
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                pagenameTextView.setText("Please capture an image");
            }
        });
    }

    ActivityResultLauncher<Intent> launchImageActivity = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if(result.getResultCode() == Activity.RESULT_OK){
                    Intent data = result.getData();
                    if(data != null && data.getData() != null){
                        selectedImageUri = data.getData();
                        selectedImagePath = selectedImageUri.getPath();
                        try{
                            selectedImageBitmap = MediaStore.Images.Media.getBitmap(
                                    this.getContentResolver(), selectedImageUri);
                            imageView.setImageBitmap(selectedImageBitmap);
                            nextButton.setEnabled(true);
                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
    );

    ActivityResultLauncher<Intent> startCamera = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {
                    if (result.getResultCode() == RESULT_OK) {
                        imageView.setImageURI(selectedImageUri);
                        nextButton.setEnabled(true);
                    }
                }
            });
    private void openCamera(View view) {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        File photoFile = null;
        try {
            photoFile = createImageFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if(photoFile != null){
            selectedImageUri = FileProvider.getUriForFile(this, "com.example.android.fileprovider", photoFile);
            cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, selectedImageUri);
            startCamera.launch(cameraIntent);
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

    }

    ActivityResultLauncher<Intent> launchAnotherActivity = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {

                }
            }
    );

    private void nextPage(View view){
        Intent myIntent = new Intent(MainActivity.this, CategoryActivity.class);
        myIntent.putExtra("BitmapImage", selectedImageUri);
        launchAnotherActivity.launch(myIntent);
    }

    String currentPhotoPath;

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

}