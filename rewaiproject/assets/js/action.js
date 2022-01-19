var inputImageBtn = document.getElementById("inputImageBtn");
let outImageMaskCamisa = document.getElementById("outImageMaskCamisa");
let outStoreCamisa = document.getElementById("outStoreCamisa");
let outImageMaskPantalon = document.getElementById("outImageMaskPantalon");
let outStorePantalon = document.getElementById("outStorePantalon");

let data = []
let count = {
    cardboard: 0,
    glass: 0,
    metal: 0,
    paper: 0,
    plastic: 0,
    trash: 0
}

inputImageBtn.onchange = evt => {
    const [file] = inputImageBtn.files
    if (file) {
        inputImage.src = URL.createObjectURL(file)
        form = new FormData
        console.log(file)
        form.append('imagen',file)
        postData('http://127.0.0.1:8000/api/testImage', form)
        .then(data => {
          console.log(data); // JSON data parsed by `data.json()` call
          console.log('resultcamisa',data.blusa)
          console.log(inputImage,outImageMaskCamisa,outImageMaskPantalon)
          outImageMaskCamisa.src = "http://127.0.0.1:8000/static/"+data.blusa          
          outImageMaskPantalon.src = "http://127.0.0.1:8000/static/"+data.pantalon
          outStoreCamisa.innerHTML = data.store_camisa
          outStorePantalon.innerHTML = data.store_pantalon
          });
    }
}  

async function postData(url = '', data = {}) {
    const response = await fetch(url, {
      method: 'POST', 
      body: data // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects
  }
  
