const video = document.getElementById("video");

// Start camera
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        alert("Camera access denied or not available");
    });

function registerStudent() {
    const studentId = document.getElementById("student_id").value.trim();
    const studentName = document.getElementById("student_name").value.trim();
    const result = document.getElementById("result");

    if (!studentId || !studentName) {
        alert("Please enter both Student ID and Student Name");
        return;
    }

    // Capture frame
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL("image/jpeg");

    // Send to backend
    fetch("/register_student", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            student_id: studentId,
            student_name: studentName,
            image: imageData
        })
    })
    .then(res => res.json())
    .then(data => {
        result.innerText = data.message;
        result.style.color = data.message.includes("successfully") ? "green" : "red";
    })
    .catch(err => {
        result.innerText = "Error registering student ❌";
        result.style.color = "red";
    });
}
