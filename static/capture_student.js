const video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(() => {
        document.getElementById("status").innerText =
            "Camera access denied";
    });

function markAttendance() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL("image/jpeg");

    fetch("/mark_student_attendance", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ image: imageData })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerText = data.message;
    });
}
