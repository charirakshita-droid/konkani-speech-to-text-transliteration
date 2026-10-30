import javax.sound.sampled.*;
import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Duration;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

public class KonkaniLiveSpeechApp extends JFrame {
    private final JTextArea devanagari = new JTextArea(5, 45);
    private final JTextArea roman = new JTextArea(5, 45);
    private final JLabel status = new JLabel("Ready. Start recording and speak in Konkani.");
    private final JButton startButton = new JButton("Start recording");
    private final JButton stopButton = new JButton("Stop & transcribe");
    private TargetDataLine microphone;
    private Thread recordingThread;
    private final AtomicBoolean recording = new AtomicBoolean(false);
    private ByteArrayOutputStream recordedBytes;
    private AudioFormat format;
    private Path wavFile;

    public KonkaniLiveSpeechApp() {
        super("Konkani Speech-to-Text — Live Microphone");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(760, 600);
        setLocationRelativeTo(null);

        devanagari.setLineWrap(true);
        devanagari.setWrapStyleWord(true);
        roman.setLineWrap(true);
        roman.setWrapStyleWord(true);
        devanagari.setEditable(false);
        roman.setEditable(false);

        stopButton.setEnabled(false);
        startButton.addActionListener(e -> startRecording());
        stopButton.addActionListener(e -> stopAndTranscribe());

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.LEFT));
        controls.add(startButton);
        controls.add(stopButton);
        controls.add(status);

        JPanel outputs = new JPanel(new GridLayout(2, 1, 8, 8));
        outputs.add(labeled("Devanagari Konkani", devanagari));
        outputs.add(labeled("Roman Konkani (machine transliteration)", roman));

        add(controls, BorderLayout.NORTH);
        add(outputs, BorderLayout.CENTER);
        add(new JLabel("Local backend: http://127.0.0.1:8000  |  Speak clearly in a quiet room."),
                BorderLayout.SOUTH);
    }

    private JPanel labeled(String title, JTextArea area) {
        JPanel panel = new JPanel(new BorderLayout(4, 4));
        panel.add(new JLabel(title), BorderLayout.NORTH);
        panel.add(new JScrollPane(area), BorderLayout.CENTER);
        return panel;
    }

    private void startRecording() {
        try {
            // Record at a commonly supported microphone format; backend resamples to 16 kHz
            // mono.
            format = new AudioFormat(44100.0f, 16, 1, true, false);
            DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
            microphone = (TargetDataLine) AudioSystem.getLine(info);
            microphone.open(format);
            microphone.start();
            recordedBytes = new ByteArrayOutputStream();
            recording.set(true);

            startButton.setEnabled(false);
            stopButton.setEnabled(true);
            devanagari.setText("");
            roman.setText("");
            status.setText("Recording... speak now, then press Stop & transcribe.");

            recordingThread = new Thread(() -> {
                byte[] buffer = new byte[4096];
                while (recording.get()) {
                    int count = microphone.read(buffer, 0, buffer.length);
                    if (count > 0) {
                        synchronized (recordedBytes) {
                            recordedBytes.write(buffer, 0, count);
                        }
                    }
                }
            }, "microphone-recorder");
            recordingThread.start();
        } catch (Exception ex) {
            status.setText("Microphone error.");
            JOptionPane.showMessageDialog(this,
                    "Could not start microphone recording.\n" + ex.getMessage()
                            + "\nCheck Windows microphone permissions and the selected input device.",
                    "Microphone error", JOptionPane.ERROR_MESSAGE);
            startButton.setEnabled(true);
            stopButton.setEnabled(false);
        }
    }

    private void stopAndTranscribe() {
        if (!recording.get())
            return;
        // Fix: tell the recording thread to finish and wait for it BEFORE closing the
        // line.
        recording.set(false);
        try {
            if (recordingThread != null)
                recordingThread.join(1500);
            microphone.stop();
            microphone.close();

            byte[] audioBytes;
            synchronized (recordedBytes) {
                audioBytes = recordedBytes.toByteArray();
            }
            if (audioBytes.length < 2048) {
                status.setText("Recording too short. Try again.");
                startButton.setEnabled(true);
                stopButton.setEnabled(false);
                return;
            }

            wavFile = Files.createTempFile("konkani-mic-", ".wav");
            try (ByteArrayInputStream bais = new ByteArrayInputStream(audioBytes);
                    AudioInputStream ais = new AudioInputStream(
                            bais, format, audioBytes.length / format.getFrameSize())) {
                AudioSystem.write(ais, AudioFileFormat.Type.WAVE, wavFile.toFile());
            }

            startButton.setEnabled(true);
            stopButton.setEnabled(false);
            status.setText("Transcribing... first run may take longer.");
            sendToBackend(wavFile);
        } catch (Exception ex) {
            status.setText("Could not prepare recording.");
            startButton.setEnabled(true);
            stopButton.setEnabled(false);
            JOptionPane.showMessageDialog(this, ex.toString(), "Recording error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void sendToBackend(Path audioFile) {
        new SwingWorker<String, Void>() {
            @Override
            protected String doInBackground() throws Exception {
                String boundary = "----JavaBoundary" + System.currentTimeMillis();
                byte[] audio = Files.readAllBytes(audioFile);
                ByteArrayOutputStream body = new ByteArrayOutputStream();
                write(body, "--" + boundary + "\r\n");
                write(body, "Content-Disposition: form-data; name=\"audio\"; filename=\"recording.wav\"\r\n");
                write(body, "Content-Type: audio/wav\r\n\r\n");
                body.write(audio);
                write(body, "\r\n--" + boundary + "--\r\n");

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create("http://127.0.0.1:8000/transcribe"))
                        .timeout(Duration.ofSeconds(180)) // first request can be slow (model warm-up)
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(body.toByteArray()))
                        .build();

                HttpClient client = HttpClient.newBuilder()
                        .version(HttpClient.Version.HTTP_1_1) // avoids HTTP/2 upgrade, which uvicorn mishandles
                        .connectTimeout(Duration.ofSeconds(10))
                        .build();
                HttpResponse<String> response = client.send(
                        request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
                if (response.statusCode() < 200 || response.statusCode() >= 300) {
                    throw new IOException("Backend returned HTTP " + response.statusCode() + ":\n" + response.body());
                }
                return response.body();
            }

            @Override
            protected void done() {
                try {
                    String json = get();
                    devanagari.setText(jsonValue(json, "devanagari"));
                    roman.setText(jsonValue(json, "roman"));
                    status.setText("Done. Please check spelling and transliteration.");
                } catch (Exception ex) {
                    status.setText("Transcription failed.");
                    Throwable cause = (ex instanceof ExecutionException && ex.getCause() != null)
                            ? ex.getCause()
                            : ex;
                    String message;
                    if (cause instanceof ConnectException) {
                        message = "Cannot reach the Python backend at http://127.0.0.1:8000.\n"
                                + "Start it first (py\\run.bat) and wait for \"Model loaded.\"";
                    } else {
                        message = "Could not transcribe.\n\n" + cause.getMessage();
                    }
                    JOptionPane.showMessageDialog(KonkaniLiveSpeechApp.this,
                            message, "Transcription error", JOptionPane.ERROR_MESSAGE);
                } finally {
                    try {
                        Files.deleteIfExists(audioFile);
                    } catch (IOException ignored) {
                    }
                }
            }
        }.execute();
    }

    private static void write(OutputStream out, String text) throws IOException {
        out.write(text.getBytes(StandardCharsets.UTF_8));
    }

    // Minimal JSON string parser for this API response; handles JSON escapes and
    // Unicode escapes.
    private String jsonValue(String json, String key) {
        String marker = "\"" + key + "\"";
        int i = json.indexOf(marker);
        if (i < 0)
            return "";
        i = json.indexOf(':', i + marker.length()) + 1;
        while (i < json.length() && Character.isWhitespace(json.charAt(i)))
            i++;
        if (i >= json.length() || json.charAt(i) != '"')
            return "";
        i++;
        StringBuilder out = new StringBuilder();
        boolean escape = false;
        for (; i < json.length(); i++) {
            char c = json.charAt(i);
            if (escape) {
                switch (c) {
                    case 'n':
                        out.append('\n');
                        break;
                    case 'r':
                        out.append('\r');
                        break;
                    case 't':
                        out.append('\t');
                        break;
                    case 'b':
                        out.append('\b');
                        break;
                    case 'f':
                        out.append('\f');
                        break;
                    case 'u':
                        if (i + 4 < json.length()) {
                            try {
                                out.append((char) Integer.parseInt(json.substring(i + 1, i + 5), 16));
                                i += 4;
                            } catch (NumberFormatException ex) {
                                out.append('u');
                            }
                        }
                        break;
                    default:
                        out.append(c);
                }
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                return out.toString();
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new KonkaniLiveSpeechApp().setVisible(true));
    }
}