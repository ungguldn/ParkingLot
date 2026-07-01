import gradio as gr
from detector import detect_video

# =====================================
# Fungsi yang dipanggil Gradio
# =====================================

def proses(video):

    if video is None:
        return None, 0, 0

    hasil_video, kosong, terisi = detect_video(video)

    return hasil_video, kosong, terisi


# =====================================
# Tampilan GUI
# =====================================

with gr.Blocks(
    title="Smart Parking Detection",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🚗 Smart Parking Detection System

        ### Project Computer Vision

        Upload video area parkir kemudian klik **Mulai Deteksi**.
        Sistem akan menentukan slot parkir **Kosong** dan **Terisi** menggunakan model Machine Learning.
        """
    )

    with gr.Row():

        video_input = gr.Video(
            label="📤 Upload Video",
            sources=["upload"],
            height=400
        )

        video_output = gr.Video(
            label="🎥 Hasil Deteksi",
            height=400
        )

    with gr.Row():

        jumlah_kosong = gr.Number(
            label="🟢 Slot Kosong",
            precision=0
        )

        jumlah_terisi = gr.Number(
            label="🔴 Slot Terisi",
            precision=0
        )

    with gr.Row():

        tombol = gr.Button(
            "🚀 Mulai Deteksi",
            variant="primary",
            size="lg"
        )

    tombol.click(
        fn=proses,
        inputs=video_input,
        outputs=[
            video_output,
            jumlah_kosong,
            jumlah_terisi
        ]
    )

    gr.Markdown(
        """
        ---
        Dibuat untuk Project Mata Kuliah Computer Vision
        """
    )

demo.launch(
    share=False,
    debug=True
)
