//var inputImg =  document.getElementById("inputImage");
var inputImg = document.getElementById("id_imagen");
var inputImageBtn = document.getElementById("inputImageBtn");
var imageName = document.getElementById("image-name");
var imgs = document.getElementById("imgs");

inputImageBtn

const cardboard = document.getElementById("cardboard");
const glass = document.getElementById("glass");
const metal = document.getElementById("metal");
const paper = document.getElementById("paper");
const plastic = document.getElementById("plastic");
const trash = document.getElementById("trash");
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
        form.append('imagen',file)
        postData('http://127.0.0.1:8000/api/testImage', form)
        .then(data => {
          console.log(data); // JSON data parsed by `data.json()` call
        });
    }
}  

async function postData(url = '', data = {}) {
    console.log('insidepost', data)
    const response = await fetch(url, {
      method: 'POST', 
      body: data // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects
  }
  
