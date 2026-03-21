async function uploadFile() {
    const file = document.getElementById("fileInput").files[0];
    const mode = document.getElementById("mode").value;
    const resultDiv = document.getElementById("result");

    if (!file) {
        alert("Select a file");
        return;
    }

    resultDiv.innerHTML = "⏳ Processing...";

    const formData = new FormData();
    formData.append("file", file);

    const endpoint = mode === "image"
        ? "http://127.0.0.1:5000/detect/image"
        : "http://127.0.0.1:5000/detect/video";

    const res = await fetch(endpoint, {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    resultDiv.innerHTML = JSON.stringify(data, null, 2);
}