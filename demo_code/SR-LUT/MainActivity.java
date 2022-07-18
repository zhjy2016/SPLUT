package com.example.sr_lut;

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
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private ImageView iv_lr;

    private ImageView iv_sr;

    private TextView tv_lr;

    private TextView tv_save;

    private TextView tv_sr;

    private void createOutputDir() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(Environment.getExternalStorageDirectory().getAbsolutePath());
        stringBuilder.append("/SRLUT");
        File file = new File(stringBuilder.toString());
        if (!file.exists())
            file.mkdirs();
    }

    public static IntStream mapping(int f1, int f2, int f3, int f4, int f5, int i1, int i2, int i3, int i4, int i5, int[] lut){
        int[] out = new int[16];
        for (int i = 0; i < 16; i++){
            out[i] = f1*lut[i1*16+i]+f2*lut[i2*16+i]+f3*lut[i3*16+i]+f4*lut[i4*16+i]+f5*lut[i5*16+i];
        }
        return Arrays.stream(out);
    }

    public static int location(int xx, int wr, int w, int N){
        int x = xx%N;
        int row = x/wr;
        int col = x%wr;
        int boxcol = col/4;
        int boxrow = row/4;
        int inside_idx = row%4*4+col%4;
        return xx/N*N+(boxrow*w+boxcol)*16+(inside_idx);
    }

    public static IntStream core2(int[] lut, int ax, int bx, int cx, int dx, int q, int L, int L2, int L3){
        int fax = ax%q;
        int fbx = bx%q;
        int fcx = cx%q;
        int fdx = dx%q;

        int iax = ax/q*L3;
        int ibx = bx/q*L2;
        int icx = cx/q*L;
        int idx = dx/q;
        int p0000 = iax + ibx + icx + idx;
        int p0001 = p0000+1;
        int p0010 = p0000+L;
        int p0011 = p0010+1;
        int p0100 = p0000+L2;
        int p0101 = p0100+1;
        int p0110 = p0100+L;
        int p0111 = p0110+1;

        int p1000 = p0000+L3;
        int p1001 = p1000+1;
        int p1010 = p1000+L;
        int p1011 = p1010+1;
        int p1100 = p1000+L2;
        int p1101 = p1100+1;
        int p1110 = p1100+L;
        int p1111 = p1110+1;

        if (fax > fbx){
            if (fbx > fcx){
                if (fcx > fdx){
                    return mapping((q-fax), (fax-fbx), (fbx-fcx), (fcx-fdx), fdx, p0000, p1000, p1100, p1110, p1111, lut);
                }else if (fbx > fdx){
                    return mapping((q-fax), (fax-fbx), (fbx-fdx), (fdx-fcx), fcx, p0000, p1000, p1100, p1101, p1111, lut);
                }else if (fax > fdx){
                    return mapping((q-fax), (fax-fdx), (fdx-fbx), (fbx-fcx), fcx, p0000, p1000, p1001, p1101, p1111, lut);
                }else {
                    return mapping((q-fdx), (fdx-fax), (fax-fbx), (fbx-fcx), fcx, p0000, p0001, p1001, p1101, p1111, lut);
                }
            }else if (fax > fcx){
                if (fbx > fdx){
                    return mapping((q-fax), (fax-fcx), (fcx-fbx), (fbx-fdx), fdx, p0000, p1000, p1010, p1110, p1111, lut);
                }else if (fcx > fdx){
                    return mapping((q-fax), (fax-fcx), (fcx-fdx), (fdx-fbx), fbx, p0000, p1000, p1010, p1011, p1111, lut);
                }else if (fax > fdx){
                    return mapping((q-fax), (fax-fdx), (fdx-fcx), (fcx-fbx), fbx, p0000, p1000, p1001, p1011, p1111, lut);
                }else {
                    return mapping((q-fdx), (fdx-fax), (fax-fcx), (fcx-fbx), fbx, p0000, p0001, p1001, p1011, p1111, lut);
                }
            }else{
                if (fbx > fdx){
                    return mapping((q-fcx), (fcx-fax), (fax-fbx), (fbx-fdx), fdx, p0000, p0010, p1010, p1110, p1111, lut);
                }else if (fcx > fdx){
                    return mapping((q-fcx), (fcx-fax), (fax-fdx), (fdx-fbx), fbx, p0000, p0010, p1010, p1011, p1111, lut);
                }else if (fax > fdx){
                    return mapping((q-fcx), (fcx-fdx), (fdx-fax), (fax-fbx), fbx, p0000, p0010, p0011, p1011, p1111, lut);
                }else {
                    return mapping((q-fdx), (fdx-fcx), (fcx-fax), (fax-fbx), fbx, p0000, p0001, p0011, p1011, p1111, lut);
                }
            }
        }else{
            if (fax > fcx){
                if (fcx > fdx){
                    return mapping((q-fbx), (fbx-fax), (fax-fcx), (fcx-fdx), fdx, p0000, p0100, p1100, p1110, p1111, lut);
                }else if (fax > fdx){
                    return mapping((q-fbx), (fbx-fax), (fax-fdx), (fdx-fcx), fcx, p0000, p0100, p1100, p1101, p1111, lut);
                }else if (fbx > fdx){
                    return mapping((q-fbx), (fbx-fdx), (fdx-fax), (fax-fcx), fcx, p0000, p0100, p0101, p1101, p1111, lut);
                }else{
                    return mapping((q-fdx), (fdx-fbx), (fbx-fax), (fax-fcx), fcx, p0000, p0001, p0101, p1101, p1111, lut);
                }
            }else if (fbx > fcx){
                if (fax > fdx){
                    return mapping((q-fbx), (fbx-fcx), (fcx-fax), (fax-fdx), fdx, p0000, p0100, p0110, p1110, p1111, lut);
                }else if (fcx > fdx){
                    return mapping((q-fbx), (fbx-fcx), (fcx-fdx), (fdx-fax), fax, p0000, p0100, p0110, p0111, p1111, lut);
                }else if (fbx > fdx){
                    return mapping((q-fbx), (fbx-fdx), (fdx-fcx), (fcx-fax), fax, p0000, p0100, p0101, p0111, p1111, lut);
                }else{
                    return mapping((q-fdx), (fdx-fbx), (fbx-fcx), (fcx-fax), fax, p0000, p0001, p0101, p0111, p1111, lut);
                }
            }else{
                if (fax > fdx){
                    return mapping((q-fcx), (fcx-fbx), (fbx-fax), (fax-fdx), fdx, p0000, p0010, p0110, p1110, p1111, lut);
                }else if (fbx > fdx){
                    return mapping((q-fcx), (fcx-fbx), (fbx-fdx), (fdx-fax), fax, p0000, p0010, p0110, p0111, p1111, lut);
                }else if (fcx > fdx){
                    return mapping((q-fcx), (fcx-fdx), (fdx-fbx), (fbx-fax), fax, p0000, p0010, p0011, p0111, p1111, lut);
                }else{
                    return mapping((q-fdx), (fdx-fcx), (fcx-fbx), (fbx-fax), fax, p0000, p0001, p0011, p0111, p1111, lut);
                }
            }
        }
    }

    public static int toRGB(int[] img, int x, int n){
        int xx = x%n;
        if (x/n == 0) {
            return img[xx] >> 16 & 0xff;
        }else if (x/n == 1){
            return img[xx] >> 8 & 0xff;
        }else{
            return img[xx] & 0xff;
        }
    }

    public static int img_fb_pos(int x, int w, int h, int n){
        if (x%w == w-1){
            return x-1;
        }else{
            return x+1;
        }
    }

    public static int img_fc_pos(int xx, int w, int h, int n){
        if (xx%n/w == h-1){
            return xx-w;
        }else{
            return xx+w;
        }
    }

    public static int img_fd_pos(int xx, int w, int h, int n){
        if (xx%w == w-1 && xx%n/w == h-1){
            return xx-w-1;
        }else if (xx%w == w-1) {
            return xx + w - 1;
        }else if(xx%n/w == h-1) {
            return xx-w+1;
        }
        else{
            return xx+w+1;
        }
    }

    public static int[] fsi2(int[] img_int, int[] lut, int n, int q, int[] b_pos, int[] c_pos, int[] d_pos){
        int L = q+1;
        int L2 = L*L;
        int L3 = L2*L;
        int[] rgb = IntStream.range(0, n*3).parallel().map(x -> toRGB(img_int, x, n)).toArray();

        int[] out = IntStream.range(0, n*3).parallel().flatMap(x -> core2(lut, rgb[x], rgb[b_pos[x]], rgb[c_pos[x]], rgb[d_pos[x]], q, L, L2, L3)).toArray();
        return out;
    }

    public static int process(int a, int b, int c, int d){
        float tmp = ((float)a + (float)b + (float)c + (float)d)/255f/16f;
        tmp = Math.max(Math.min(tmp, 1f), 0f)*255f;
        return Math.round(tmp);
    }

    public static int rotate_pos(int xx, int w, int h, int N, int angel){
        int x = xx%N;
        int y;
        if(angel == 90){
            y = (h-1-x%h)*w+x/h;
        }else if(angel == 180){
            y = w*(h-1-x/w)+w-1-x%w;
        }else if(angel == 270){
            y = x%h*w+w-1-x/h;
        }else{
            y = x;
        }
        return y+xx/N*N;
    }

    public static int rotate_pos1(int x, int w, int h, int angel){
        if(angel == 90){
            return (h-1-x%h)*w+x/h;
        }else if(angel == 180){
            return w*(h-1-x/w)+w-1-x%w;
        }else if(angel == 270){
            return x%h*w+w-1-x/h;
        }else{
            return x;
        }
    }

    private void doSRLUT(String paramString, Bitmap bitmap) {
        try {
            InputStreamReader isr = new InputStreamReader(getAssets().open("dict_v150_G1_uint8_4bit.csv"));
            BufferedReader br = new BufferedReader(isr);
            List<String> list = new ArrayList<>();

            String line = null;
            while ((line = br.readLine()) != null){
                list.add(line);
            }
            int[] lut = list.stream().mapToInt(num -> Integer.parseInt(num)).toArray();


            int w = bitmap.getWidth();
            int h = bitmap.getHeight();
            StringBuilder str1 = new StringBuilder();
            str1.append("LR input: ");
            str1.append(paramString);
            str1.append("; Size: ");
            str1.append(w);
            str1.append("*");
            str1.append(h);
            int N = 16 * w * h;
            int wr = 4 * w;
            int hr = 4 * h;
            int[] bi = new int[w * h];
            int Nr = w*h;
            bitmap.getPixels(bi, 0, w, 0, 0, w, h);

            long l = System.nanoTime();

            IntStream bi0 = IntStream.range(0, Nr).parallel();
            IntStream bi1 = IntStream.range(0, Nr).parallel();
            IntStream bi2 = IntStream.range(0, Nr).parallel();
            IntStream bi3 = IntStream.range(0, Nr).parallel();
            IntStream r0 = IntStream.range(0, 3*N).parallel();
            IntStream r1 = IntStream.range(0, 3*N).parallel();
            IntStream r2 = IntStream.range(0, 3*N).parallel();
            IntStream r3 = IntStream.range(0, 3*N).parallel();
            IntStream result = IntStream.range(0, N).parallel();
            Bitmap bitmap_r = Bitmap.createBitmap(w * 4, h * 4, Bitmap.Config.ARGB_8888);
            IntStream out_t = IntStream.range(0, 3 * N).parallel();

            int[] pi0 = bi0.map(x -> bi[rotate_pos1(x, w, h, 0)]).toArray();
            int[] pi1 = bi1.map(x -> bi[rotate_pos1(x, w, h, 90)]).toArray();
            int[] pi2 = bi2.map(x -> bi[rotate_pos1(x, w, h, 180)]).toArray();
            int[] pi3 = bi3.map(x -> bi[rotate_pos1(x, w, h, 270)]).toArray();

            long l1 = System.nanoTime();

            int q = 16;
            int[] b_pos0 = IntStream.range(0, Nr*3).parallel().map(x -> img_fb_pos(x, w, h, Nr)).toArray();
            int[] c_pos0 = IntStream.range(0, Nr*3).parallel().map(x -> img_fc_pos(x, w, h, Nr)).toArray();
            int[] d_pos0 = IntStream.range(0, Nr*3).parallel().map(x -> img_fd_pos(x, w, h, Nr)).toArray();
            int[] b_pos1 = IntStream.range(0, Nr*3).parallel().map(x -> img_fb_pos(x, h, w, Nr)).toArray();
            int[] c_pos1 = IntStream.range(0, Nr*3).parallel().map(x -> img_fc_pos(x, h, w, Nr)).toArray();
            int[] d_pos1 = IntStream.range(0, Nr*3).parallel().map(x -> img_fd_pos(x, h, w, Nr)).toArray();
            int[] out0 = fsi2(pi0, lut, Nr, q, b_pos0, c_pos0, d_pos0);
            int[] out1 = fsi2(pi1, lut, Nr, q, b_pos1, c_pos1, d_pos1);
            int[] out2 = fsi2(pi2, lut, Nr, q, b_pos0, c_pos0, d_pos0);
            int[] out3 = fsi2(pi3, lut, Nr, q, b_pos1, c_pos1, d_pos1);

            int[] loc0 = IntStream.range(0, 3 * N).parallel().map(x -> location(x, wr, w, N)).toArray();
            int[] loc1 = IntStream.range(0, 3 * N).parallel().map(x -> location(x, hr, h, N)).toArray();
            int[] out0r = r0.map(x -> out0[loc0[rotate_pos(x, wr, hr, N, 0)]]).toArray();
            int[] out1r = r1.map(x -> out1[loc1[rotate_pos(x, hr, wr, N, 270)]]).toArray();
            int[] out2r = r2.map(x -> out2[loc0[rotate_pos(x, wr, hr, N, 180)]]).toArray();
            int[] out3r = r3.map(x -> out3[loc1[rotate_pos(x, hr, wr, N, 90)]]).toArray();


            int[] out = out_t.map(x -> process(out0r[x], out1r[x], out2r[x], out3r[x])).toArray();

            bitmap_r.setPixels(result.map(x -> (255 << 24 | (out[x]) << 16 | (out[N + x]) << 8 | (out[2 * N + x]))).toArray(), 0, w * 4, 0, 0, w * 4, h * 4);
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
        if (ContextCompat.checkSelfPermission((Context)paramActivity, "android.permission.WRITE_EXTERNAL_STORAGE") == 0) {
            bool = true;
        } else {
            bool = false;
        }
        if (!bool) {
            ActivityCompat.requestPermissions(paramActivity, new String[] { "android.permission.WRITE_EXTERNAL_STORAGE" }, 112);
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
                doSRLUT(getFileName(uri), bitmap);
                iv_lr.setImageBitmap(bitmap);
            } catch (IOException iOException) {
                Toast.makeText((Context)this, "Image load failed", 1).show();
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        requestPermission((Activity)this);

        this.iv_lr = (ImageView)findViewById(R.id.ivlr);
        this.iv_sr = (ImageView)findViewById(R.id.ivsr);
        this.tv_lr = (TextView)findViewById(R.id.tvlr);
        this.tv_sr = (TextView)findViewById(R.id.tvsr);
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
        Spinner spinner = (Spinner)findViewById(R.id.spinner);
        ArrayAdapter arrayAdapter = new ArrayAdapter((Context)this, 17367049, (Object[])items);
        arrayAdapter.setDropDownViewResource(17367049);
        spinner.setAdapter((SpinnerAdapter)arrayAdapter);
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
                        MainActivity.this.doSRLUT(items[param1Int], bitmap);
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