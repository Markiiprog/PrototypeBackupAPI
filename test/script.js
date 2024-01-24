async function uploadAudio() {
    const fileInput = document.getElementById('audioInput');
    await uploadFile(fileInput, 'http://192.168.0.2:8000/transcribe/audio', 'audioResult');
}

async function uploadVideo() {
    const fileInput = document.getElementById('videoInput');
    await uploadFile(fileInput, 'http://192.168.0.2:8000/transcribe/video', 'videoResult');
}

async function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    await uploadFile(fileInput, 'http://192.168.0.2:8000/transcribe/image', 'imageResult');
}


async function uploadFile(fileInput, endpoint, resultDivId) {
    const file = fileInput.files[0];

    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

        
            const data = await response.json();
        
            if (data && data.Transcription !== undefined) {
                // Display the result
                const resultDiv = document.getElementById(resultDivId);
                resultDiv.innerHTML = `<p>Transcription: ${data.Transcription}</p>`;
                resultDiv.innerHTML += `<p>Transcription to Braille: ${data.Braille}</p>`;
            } else {
                console.error('Unexpected API response format:', data);
            }

        } catch (error) {
            console.error('Error uploading file:', error);
        }
    } else {
        console.error(`Please select a file`);
    }
}
