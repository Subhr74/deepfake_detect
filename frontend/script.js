const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultDiv = document.getElementById("result");

let selectedFile;

// Drag & Drop
dropArea.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
    selectedFile = e.target.files[0];
    showImage(selectedFile);
};

dropArea.ondragover = (e) => {
    e.preventDefault();
};

dropArea.ondrop = (e) => {
    e.preventDefault();
    selectedFile = e.dataTransfer.files[0];
    showImage(selectedFile);
};

function showImage(file) {
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
    };
    img.src = URL.createObjectURL(file);
}

async function uploadFile() {
    if (!selectedFile) {
        alert("Select an image");
        return;
    }

    resultDiv.innerHTML = "⏳ Processing...";

    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch("http://127.0.0.1:5000/detect", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    drawResults(data);
}

function drawResults(data) {
    resultDiv.innerHTML = "";

    if (!data.faces) {
        resultDiv.innerHTML = "No faces detected";
        return;
    }

    data.faces.forEach(face => {
        const [x, y, w, h] = face.box;

        ctx.strokeStyle = face.label === "Real" ? "green" : "red";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "white";
        ctx.fillText(
            `${face.label} (${face.confidence}%)`,
            x,
            y - 10
        );

        const div = document.createElement("div");
        div.className = `result-box ${face.label.toLowerCase()}`;
        div.innerText = `${face.label} - ${face.confidence}%`;

        resultDiv.appendChild(div);
    });
}