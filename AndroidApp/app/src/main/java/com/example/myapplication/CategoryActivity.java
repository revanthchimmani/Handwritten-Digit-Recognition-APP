package com.example.myapplication;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.FileUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.time.LocalDateTime;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class CategoryActivity extends AppCompatActivity {

    Spinner category;
    Button submit;
    Uri selectedImageUri;
    String categorySelected;
    File image = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_category);

        Bundle bundle = getIntent().getExtras();
        selectedImageUri = (Uri) bundle.get("BitmapImage");

        InputStream inputStream;
        OutputStream outputStream;

        try {
            inputStream = getApplicationContext().getContentResolver().openInputStream(selectedImageUri);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                image = new File(getApplicationContext().getCacheDir(), LocalDateTime.now() + ".jpg");
            }
            image.createNewFile();
            outputStream = new FileOutputStream(image);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                FileUtils.copy(inputStream, outputStream);
                outputStream.flush();
            }
        }catch (Exception e){

        }

        ImageView imageView = findViewById(R.id.imageToSubmit);
        Bitmap bitmap = BitmapFactory.decodeFile(image.getAbsolutePath());
        imageView.setImageBitmap(bitmap);

        submit = findViewById(R.id.submit);
        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                sendImageAndCategory();

            }
        });
    }

    ActivityResultLauncher<Intent> launchAnotherActivity = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {

                }
            }
    );

    private void sendImageAndCategory(){

        RequestBody requestBody = new MultipartBody.Builder().setType(MultipartBody.FORM).addFormDataPart(
                "image",
                image.getName(),
                RequestBody.create(image, MediaType.parse("image/*jpg"))).build();
        Request request = new Request.Builder().url(getString(R.string.ip_address) + "predict").post(requestBody).build();
        OkHttpClient okHttpClient = new OkHttpClient();

        okHttpClient.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(getApplicationContext(), "Failure connecting to server", Toast.LENGTH_LONG).show();
                    }
                });
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        Log.e("Api Response", String.valueOf(response));
                        Toast.makeText(getApplicationContext(), "Image uploaded", Toast.LENGTH_LONG).show();
                        Intent changeActivityIntent = new Intent(CategoryActivity.this, SuccessScreenActivity.class);
                       launchAnotherActivity.launch(changeActivityIntent);
                    }
                });
            }
        });
    }


}