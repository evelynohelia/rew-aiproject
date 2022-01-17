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


document.addEventListener('DOMContentLoaded', () => loadDataImg())

inputImg.addEventListener("change", (e) => {
    imageName.innerHTML = e.target.files[0].name;
    var image = document.getElementById('output');
    image.src = URL.createObjectURL(e.target.files[0]);
});

inputImageBtn.addEventListener("click", () => {
    document.getElementById("id_imagen").click();
    var result = document.getElementById("result");
    var contName = document.getElementById("cont-imagen-name");
    result.style.display = "none";
    result.style.visibility = "hidden";
    contName.style.visibility = "visible";
    contName.style.display = "block";
    //document.getElementById("inputImage").click();
});

var settings = {
    "async": true,
    "crossDomain": true,
    "url": "https://clasification-0871.restdb.io/rest/images",
    "method": "GET",
    "headers": {
        "content-type": "application/json",
        "x-apikey": "6004c9251346a1524ff12b8a",
        "cache-control": "no-cache"
    }
}


function loadDataImg() {
    $.ajax(settings).done(function (response) {
        data = response;

        for (const d in response) {
            dat = response[d];
            if (dat.class == "cardboard") {
                count.cardboard += 1;
                getHTML(dat);
            } else if (dat.class == "glass") {
                count.glass += 1;
            } else if (dat.class == "metal") {
                count.metal += 1;
            } else if (dat.class == "paper") {
                count.paper += 1;
            } else if (dat.class == "plastic") {
                count.plastic += 1
            } else if (dat.class == "trash") {
                count.trash += 1;
            }
        }
        cardboard.innerHTML = "cardboard (" + count.cardboard + ")";
        glass.innerHTML = "glass (" + count.glass + ")";
        metal.innerHTML = "metal (" + count.metal + ")";
        paper.innerHTML = "paper (" + count.paper + ")";
        plastic.innerHTML = "plastic (" + count.plastic + ")";
        trash.innerHTML = "trash (" + count.trash + ")";
        events();
    });
}

function getHTML(dat) {
    var div = document.createElement("div");
    div.className = "col-xl-3 col-md-4 col-sm-6 mb-2"
    var img = document.createElement("img");
    img.className = "img-fluid rounded";
    img.src = dat.img;
    div.appendChild(img);
    let p = document.createElement("p");
    p.innerHTML = "Predicción: " + dat.accuracy + "%" + " (" + dat.model + ")";
    div.appendChild(p);
    imgs.appendChild(div);
}

function events() {
    cardboard.addEventListener('click', () => {
        cardboard.className = "nav-link active";
        glass.className = "nav-link";
        metal.className = "nav-link";
        paper.className = "nav-link";
        plastic.className = "nav-link";
        trash.className = "nav-link";
        loadClass("cardboard");
    });
    glass.addEventListener('click', () => {
        cardboard.className = "nav-link";
        glass.className = "nav-link active";
        metal.className = "nav-link";
        paper.className = "nav-link";
        plastic.className = "nav-link";
        trash.className = "nav-link";
        loadClass("glass");
    });
    metal.addEventListener('click', () => {
        cardboard.className = "nav-link";
        glass.className = "nav-link";
        metal.className = "nav-link active";
        paper.className = "nav-link";
        plastic.className = "nav-link";
        trash.className = "nav-link";
        loadClass("metal")
    });
    paper.addEventListener('click', () => {
        cardboard.className = "nav-link";
        glass.className = "nav-link";
        metal.className = "nav-link";
        paper.className = "nav-link active";
        plastic.className = "nav-link";
        trash.className = "nav-link";
        loadClass("paper");
    });
    plastic.addEventListener('click', () => {
        cardboard.className = "nav-link";
        glass.className = "nav-link";
        metal.className = "nav-link";
        paper.className = "nav-link";
        plastic.className = "nav-link active";
        trash.className = "nav-link";
        loadClass("plastic");
    });
    trash.addEventListener('click', () => {
        cardboard.className = "nav-link";
        glass.className = "nav-link";
        metal.className = "nav-link";
        paper.className = "nav-link";
        plastic.className = "nav-link";
        trash.className = "nav-link active";
        loadClass("trash");
    });
}

function loadClass(clase) {
    imgs.innerHTML = "";
    for (const d in data) {
        let dat = data[d];
        if (clase == dat.class) {
           getHTML(dat);
        }
    }
}