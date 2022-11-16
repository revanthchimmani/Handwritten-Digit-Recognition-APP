package com.example.myapplication;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

public class SuccessScreenActivity extends AppCompatActivity {

    ActivityResultLauncher<Intent> launchAnotherActivity = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            new ActivityResultCallback<ActivityResult>() {
                @Override
                public void onActivityResult(ActivityResult result) {

                }
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_success_screen);
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Intent changeActivityIntent = new Intent(SuccessScreenActivity.this, MainActivity.class);
                launchAnotherActivity.launch(changeActivityIntent);
            }
        }, 1500);
    }
}