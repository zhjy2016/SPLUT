package com.example.splut_l;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.SpinnerAdapter;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.DoubleStream;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private ImageView iv_lr;

    private ImageView iv_sr;

    private TextView tv_lr;

    private TextView tv_save;

    private TextView tv_sr;

    private double[][][] luts;

    private void createOutputDir() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(Environment.getExternalStorageDirectory().getAbsolutePath());
        stringBuilder.append("/SRLUT");
        File file = new File(stringBuilder.toString());
        if (!file.exists())
            file.mkdirs();
    }

    public static int get_top_pos(int x, int w, int h) {
        if (x / w == 0) {
            return x + w;
        } else {
            return x - w;
        }
    }


    public static int get_bottom_pos_all(int x, int w, int h) {
        if ((x / w) % h == (h - 1)) {
            return x - w;
        } else {
            return x + w;
        }
    }


    public static int get_left_pos(int x, int w, int h) {
        if (x % w == 0) {
            return x + 1;
        } else {
            return x - 1;
        }
    }

    public static int get_right_pos(int x, int w, int h) {
        if (x % w == (w - 1)) {
            return x - 1;
        } else {
            return x + 1;
        }
    }

    public static int get_right_bottom_pos_all(int x, int w, int h, int N) {
        if ((x / w) % h == h - 1 && x % w == w - 1) {
            return x / N * N + (h - 1) * w - 2;
        } else if ((x / w) % h == h - 1) {
            return x - w + 1;
        } else if (x % w == w - 1) {
            return x + w - 1;
        } else {
            return x + w + 1;
        }
    }

    public static int clip_quan(float x) {
        return (int) (Math.floor(Math.max(Math.min(x, 7.99f), -8f)) + 8f);
    }

    public static int layer23_pos(double[] ds, double[] out1, int x, int w, int h, int L2, int L3, String str, int num) {
        int chanum = 16;
        if (str == "k221") {
            if(num == 1) {
                int top_pos = get_top_pos(x, w, h);
                int a1 = clip_quan((float) ds[x] + (float) out1[x * chanum] + (float) ds[top_pos] + (float) out1[top_pos * chanum + 2]);
                int bottom_pos = get_bottom_pos_all(x, w, h);
                int b1 = clip_quan((float) ds[bottom_pos] + (float) out1[bottom_pos * chanum] + (float) ds[x] + (float) out1[x * chanum + 2]);
                int c1 = clip_quan((float) ds[x] + (float) out1[x * chanum + 1] + (float) ds[top_pos] + (float) out1[top_pos * chanum + 3]);
                int d1 = clip_quan((float) ds[bottom_pos] + (float) out1[bottom_pos * chanum + 1] + (float) ds[x] + (float) out1[x * chanum + 3]);
                return a1 * L3 + b1 * L2 + c1 * 16 + d1;
            }else{
                int top_pos = get_top_pos(x, w, h);
                int a1 = clip_quan((float) ds[x] + (float) out1[x * chanum+4] + (float) ds[top_pos] + (float) out1[top_pos * chanum + 6]);
                int bottom_pos = get_bottom_pos_all(x, w, h);
                int b1 = clip_quan((float) ds[bottom_pos] + (float) out1[bottom_pos * chanum+4] + (float) ds[x] + (float) out1[x * chanum + 6]);
                int c1 = clip_quan((float) ds[x] + (float) out1[x * chanum + 5] + (float) ds[top_pos] + (float) out1[top_pos * chanum + 7]);
                int d1 = clip_quan((float) ds[bottom_pos] + (float) out1[bottom_pos * chanum + 5] + (float) ds[x] + (float) out1[x * chanum + 7]);
                return a1 * L3 + b1 * L2 + c1 * 16 + d1;
            }
        } else {
            if(num == 1) {
                int left_pos = get_left_pos(x, w, h);
                int a1 = clip_quan((float) ds[x] + (float) out1[x * chanum + 8] + (float) ds[left_pos] + (float) out1[left_pos * chanum + 10]);
                int right_pos = get_right_pos(x, w, h);
                int b1 = clip_quan((float) ds[right_pos] + (float) out1[right_pos * chanum + 8] + (float) ds[x] + (float) out1[x * chanum + 10]);
                int c1 = clip_quan((float) ds[x] + (float) out1[x * chanum + 9] + (float) ds[left_pos] + (float) out1[left_pos * chanum + 11]);
                int d1 = clip_quan((float) ds[right_pos] + (float) out1[right_pos * chanum + 9] + (float) ds[x] + (float) out1[x * chanum + 11]);
                return a1 * L3 + b1 * L2 + c1 * 16 + d1;
            }else{
                int left_pos = get_left_pos(x, w, h);
                int a1 = clip_quan((float) ds[x] + (float) out1[x * chanum + 12] + (float) ds[left_pos] + (float) out1[left_pos * chanum + 14]);
                int right_pos = get_right_pos(x, w, h);
                int b1 = clip_quan((float) ds[right_pos] + (float) out1[right_pos * chanum + 12] + (float) ds[x] + (float) out1[x * chanum + 14]);
                int c1 = clip_quan((float) ds[x] + (float) out1[x * chanum + 13] + (float) ds[left_pos] + (float) out1[left_pos * chanum + 15]);
                int d1 = clip_quan((float) ds[right_pos] + (float) out1[right_pos * chanum + 13] + (float) ds[x] + (float) out1[x * chanum + 15]);
                return a1 * L3 + b1 * L2 + c1 * 16 + d1;
            }
        }
    }

    public static int shuffle_all(int xx, int w, int wr, int Nr) {
        int x = xx % Nr;
        int row = x / wr;
        int col = x % wr;
        int boxcol = col / 4;
        int boxrow = row / 4;
        int inside_idx = row % 4 * 4 + col % 4;
        return xx / Nr * Nr + (boxrow * w + boxcol) * 16 + (inside_idx);
    }

    public DoubleStream lookup1(double xx, int[] isa, int w, int h, int L2, int L3, int N){
        int x = (int)xx;
        int idx = isa[x] * L3 + isa[get_right_pos(x, w, h)] * L2 + isa[get_bottom_pos_all(x, w, h)] * 16 + isa[get_right_bottom_pos_all(x, w, h, N)];
        return (x / N) % 2 == 0 ? Arrays.stream(luts[0][idx]) : Arrays.stream(luts[9][idx]);
    }

    public int process(int x, double[] out, int Nr, int Nr2, int w, int wr) {
        int idx = x/Nr*Nr2+x%Nr;
        return Math.round(Math.max(Math.min((float) out[shuffle_all(idx, w, wr, Nr)] + (float) out[shuffle_all(idx+Nr, w, wr, Nr)], 1f), 0f) * 255f);
    }

    public DoubleStream lookup2_add(double xx, double[] ds, double[] out1, int w, int h, int L2, int L3, int N){
        int x = (int)xx;
        double[] out21 = (x / N) % 2 == 0 ? luts[1][layer23_pos(ds, out1, x, w, h, L2, L3, "k221", 1)] : luts[10][layer23_pos(ds, out1, x, w, h, L2, L3, "k221", 1)];
        double[] out22 = (x / N) % 2 == 0 ? luts[2][layer23_pos(ds, out1, x, w, h, L2, L3, "k221", 2)] : luts[11][layer23_pos(ds, out1, x, w, h, L2, L3, "k221", 2)];
        double[] out23 = (x / N) % 2 == 0 ? luts[3][layer23_pos(ds, out1, x, w, h, L2, L3, "k212", 1)] : luts[12][layer23_pos(ds, out1, x, w, h, L2, L3, "k212", 1)];
        double[] out24 = (x / N) % 2 == 0 ? luts[4][layer23_pos(ds, out1, x, w, h, L2, L3, "k212", 2)] : luts[13][layer23_pos(ds, out1, x, w, h, L2, L3, "k212", 2)];
        int chanum = 16;
        double[] out2 = new double[chanum];
        for (int i = 0; i < chanum; i++){
            out2[i] = out1[x*chanum+i] + (out21[i] + out22[i] + out23[i] + out24[i]) / 4;
        }
        return Arrays.stream(out2);
    }

    public DoubleStream lookup3_add(double xx, double[] ds, double[] out2, int w, int h, int L2, int L3, int N){
        int x = (int)xx;
        double[] out31 = (x / N) % 2 == 0 ? luts[5][layer23_pos(ds, out2, x, w, h, L2, L3, "k221", 1)] : luts[14][layer23_pos(ds, out2, x, w, h, L2, L3, "k221", 1)];
        double[] out32 = (x / N) % 2 == 0 ? luts[6][layer23_pos(ds, out2, x, w, h, L2, L3, "k221", 2)] : luts[15][layer23_pos(ds, out2, x, w, h, L2, L3, "k221", 2)];
        double[] out33 = (x / N) % 2 == 0 ? luts[7][layer23_pos(ds, out2, x, w, h, L2, L3, "k212", 1)] : luts[16][layer23_pos(ds, out2, x, w, h, L2, L3, "k212", 1)];
        double[] out34 = (x / N) % 2 == 0 ? luts[8][layer23_pos(ds, out2, x, w, h, L2, L3, "k212", 2)] : luts[17][layer23_pos(ds, out2, x, w, h, L2, L3, "k212", 2)];
        int chanum = 16;
        double[] out3 = new double[chanum];
        for (int i = 0; i < chanum; i++){
            out3[i] = ds[x] + (out31[i] + out32[i] + out33[i] + out34[i]) / 4;
        }
        return Arrays.stream(out3);
    }

    public double[] lookupall(int[] isa, int w, int h, int N) {
        int[] is = Arrays.copyOf(isa, isa.length);
        double[] ds = Arrays.stream(is).asDoubleStream().map(x -> (float)x / 16f).toArray();

        int L3 = 16 * 16 * 16;
        int L2 = 16 * 16;
        double[] out1 = IntStream.range(0, 6 * N).parallel().asDoubleStream().flatMap(x -> lookup1(x, isa, w, h, L2, L3, N)).toArray(); // 8*w*h
        double[] out2 = IntStream.range(0, 6 * N).parallel().asDoubleStream().flatMap(x -> lookup2_add(x, ds, out1, w, h, L2, L3, N)).toArray();
        double[] x3_in2 = IntStream.range(0, 6 * N).parallel().asDoubleStream().flatMap(x -> lookup3_add(x, ds, out2, w, h, L2, L3, N)).toArray();
        return x3_in2;
    }

    private void doSRLUTnew(String paramString, Bitmap bitmap, double[][][] luts) {
        try{
            int w = bitmap.getWidth();
            int h = bitmap.getHeight();
            StringBuilder str1 = new StringBuilder();
            str1.append("LR input: ");
            str1.append(paramString);
            str1.append("; Size: ");
            str1.append(w);
            str1.append("*");
            str1.append(h);
            int N = w * h;
            int wr = 4 * w;
            int hr = 4 * h;
            int[] bi = new int[w * h];
            int Nr = wr * hr;
            int Nr2 = Nr * 2;
            int Nr3 = 3 * Nr;
            bitmap.getPixels(bi, 0, w, 0, 0, w, h);

            long l = System.nanoTime();
            IntStream out_t4 = IntStream.range(0, Nr).parallel();
            Bitmap bitmap_r = Bitmap.createBitmap(w * 4, h * 4, Bitmap.Config.ARGB_8888);

            int[] data = IntStream.range(0, 6 * N).parallel().map(x -> bi[x % N] >> 4 * (5 - x / N) & 0xf).toArray();

            double[] out = lookupall(data, w, h, N);

            int[] out_p = IntStream.range(0, 3 * Nr).parallel().map(x -> process(x, out, Nr, Nr2, w, wr)).toArray();

            bitmap_r.setPixels(out_t4.map(x -> (255 << 24 | (out_p[x]) << 16 | (out_p[x + Nr]) << 8 | (out_p[x + Nr2]))).toArray(), 0, w * 4, 0, 0, w * 4, h * 4);

            l = (System.nanoTime() - l) / 1000L / 1000L;

            this.iv_sr.setImageBitmap(bitmap_r);
            tv_lr.setText(str1);
            StringBuilder str2 = new StringBuilder();
            str2.append("SR runtime: ");
            str2.append(l);
            str2.append("ms; Size: ");
            str2.append(wr);
            str2.append("*");
            str2.append(hr);
            tv_sr.setText(str2);

            createOutputDir();
            String str_save1 = Environment.getExternalStorageDirectory().toString();
            StringBuilder str_save2 = new StringBuilder();
            str_save2.append("SRLUT/output_");
            str_save2.append(paramString);
            File file = new File(str_save1, str_save2.toString());
            FileOutputStream fileOutputStream = new FileOutputStream(file);
            bitmap_r.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream);
            StringBuilder str3 = new StringBuilder();
            str3.append("Saved to ");
            str3.append(file.getAbsolutePath());
            tv_save.setText(str3.toString());
        } catch (Exception e){
        }
    }

    private void requestPermission(Activity paramActivity) {
        boolean bool;
        if (ContextCompat.checkSelfPermission((Context) paramActivity, "android.permission.WRITE_EXTERNAL_STORAGE") == 0) {
            bool = true;
        } else {
            bool = false;
        }
        if (!bool) {
            ActivityCompat.requestPermissions(paramActivity, new String[]{"android.permission.WRITE_EXTERNAL_STORAGE"}, 112);
        } else {
            createOutputDir();
        }
    }

    public String getFileName(Uri paramUri) {
        boolean bool = paramUri.getScheme().equals("content");
        String str1 = null;
        String str2 = null;
        if (bool) {
            Cursor cursor = getContentResolver().query(paramUri, null, null, null, null);
            if (cursor.moveToFirst())
                str1 = cursor.getString(cursor.getColumnIndex("_display_name")); //["_id", "_display_name", "_size", "mime_type", "_data", +5 more]
            cursor.close();
        }
        str2 = str1;
        if (str1 == null) {
            String str = paramUri.getPath();
            int i = str.lastIndexOf('/');
            str2 = str;
            if (i != -1)
                str2 = str.substring(i + 1);
        }
        return str2;
    }

    public void onActivityResult(int paramInt1, int paramInt2, Intent paramIntent) {
        if (paramInt1 == 1 && paramInt2 == -1 && paramIntent != null) {
            Uri uri = paramIntent.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                doSRLUTnew(getFileName(uri), bitmap, luts);
                iv_lr.setImageBitmap(bitmap);
            } catch (IOException iOException) {
                Toast.makeText((Context) this, "Image load failed", 1).show();
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        requestPermission((Activity) this);

        try {
            int nlut = 18;
            luts = new double[nlut][65536][];
            String[] lut_names = {"LUTA1_122", "LUTA2_221_1", "LUTA2_221_2", "LUTA2_212_1", "LUTA2_212_2",
                    "LUTA3_221_1", "LUTA3_221_2", "LUTA3_212_1", "LUTA3_212_2",
                    "LUTB1_122", "LUTB2_221_1", "LUTB2_221_2", "LUTB2_212_1", "LUTB2_212_2",
                    "LUTB3_221_1", "LUTB3_221_2", "LUTB3_212_1", "LUTB3_212_2",};
            for (int ni = 0; ni < nlut; ni++) {
                String name = lut_names[ni];
                InputStreamReader a1 = new InputStreamReader(getAssets().open(name + ".csv"));
                BufferedReader ba1 = new BufferedReader(a1);
                String line = null;
                int line_idx = 0;
                while ((line = ba1.readLine()) != null) {
                    String[] stmp = line.split(",");
                    double[] dtmp = new double[stmp.length];
                    for (int i = 0; i < stmp.length; i++) {
                        dtmp[i] = Double.parseDouble(stmp[i]);
                    }
                    luts[ni][line_idx] = dtmp;
                    line_idx += 1;
                }
            }
        } catch (IOException iOException) {
            Toast.makeText((Context) MainActivity.this, "LUTs load failed", 1).show();
        }

        this.iv_lr = (ImageView) findViewById(R.id.ivlr);
        this.iv_sr = (ImageView) findViewById(R.id.ivsr);
        this.tv_lr = (TextView)findViewById(R.id.tvlr);
        this.tv_sr = (TextView) findViewById(R.id.tvsr);
        this.tv_save = (TextView)findViewById(R.id.tv_save);
        final String[] items = new String[16];
        items[0] = "Choose image";
        items[1] = "Browse image ...";
        items[2] = "baboon.png";
        items[3] = "barbara.png";
        items[4] = "bridge.png";
        items[5] = "coastguard.png";
        items[6] = "comic.png";
        items[7] = "face.png";
        items[8] = "flowers.png";
        items[9] = "foreman.png";
        items[10] = "lenna.png";
        items[11] = "man.png";
        items[12] = "monarch.png";
        items[13] = "pepper.png";
        items[14] = "ppt3.png";
        items[15] = "zebra.png";
        Spinner spinner = (Spinner) findViewById(R.id.spinner);
        ArrayAdapter arrayAdapter = new ArrayAdapter((Context) this, 17367049, (Object[]) items);
        arrayAdapter.setDropDownViewResource(17367049);
        spinner.setAdapter((SpinnerAdapter) arrayAdapter);
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            public void onItemSelected(AdapterView<?> param1AdapterView, View param1View, int param1Int, long param1Long) {
                spinner.setSelection(0);
                if (param1Int == 1) {
                    Intent intent = new Intent();
                    intent.setType("image/*");
                    intent.setAction("android.intent.action.GET_CONTENT");
                    MainActivity.this.startActivityForResult(Intent.createChooser(intent, "Select Picture"), 1);
                } else if (param1Int >= 2) {
                    AssetManager assetManager = MainActivity.this.getAssets();
                    try {
                        StringBuilder stringBuilder = new StringBuilder();
                        stringBuilder.append("Set14_LR/");
                        stringBuilder.append(items[param1Int]);
                        Bitmap bitmap = BitmapFactory.decodeStream(assetManager.open(stringBuilder.toString()));
                        MainActivity.this.doSRLUTnew(items[param1Int], bitmap, luts);
                        iv_lr.setImageBitmap(bitmap);
                    } catch (IOException iOException) {
                        Toast.makeText((Context) MainActivity.this, "Image load failed", 1).show();
                    }
                }
            }

            public void onNothingSelected(AdapterView<?> param1AdapterView) {
//                MainActivity.this.tv_lr.setText("");
            }
        });
    }
}