import gradio as gr
from detector import detect_video

def proses(video):

    hasil,kosong,terisi = detect_video(video)

    return hasil,kosong,terisi

with gr.Blocks(title="Smart Parking Detection") as demo:

    gr.Markdown("# 🚗 Smart Parking Detection System")
    gr.Markdown("Upload video area parkir untuk mendeteksi slot kosong dan terisi.")

    with gr.Row():

        video_input = gr.Video(
            label="Upload Video",
            sources=["upload"]
        )

        video_output = gr.Video(
            label="Hasil Deteksi"
        )

    with gr.Row():

        kosong = gr.Number(label="🟢 Slot Kosong")

        terisi = gr.Number(label="🔴 Slot Terisi")

    tombol = gr.Button("Mulai Deteksi")

    tombol.click(
        fn=proses,
        inputs=video_input,
        outputs=[
            video_output,
            kosong,
            terisi
        ]
    )

demo.launch()