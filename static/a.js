var canvas = document.getElementById('bg-night');

function abc() {
    var check = getEle('theme');
    var el = document.querySelectorAll('.rays');

    if (check.checked === true) {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
        for (i = 0; i < el.length; i++) {
            el[i].style.strokeDashoffset = '12';
            el[i].style.strokeDasharray = '12';
        }
        getEle('cloudy_sun').style.animationName = "slideS";
        getEle('cloudy_moon').style.animationName = "";
        getEle('sun_center').style.animationName = "fade";
        getEle('moon').style.opacity = "1";
        canvas.style.opacity = "1"
    } else {
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
        for (i = 0; i < el.length; i++) {
            el[i].style.strokeDashoffset = '0';
            el[i].style.strokeDasharray = '12';
        }
        getEle('cloudy_moon').style.animationName = "slideM";
        getEle('cloudy_sun').style.animationName = "";
        getEle('cloudy_sun').style.opacity = "0";
        getEle('sun_center').style.animationName = "";
        getEle('moon').style.opacity = "0";
        canvas.style.opacity = "0"

    }
}

var ctx = canvas.getContext("2d");

var w = window.innerWidth;
var h = window.innerHeight;

canvas.width = w;
canvas.height = h;
var angle = 0;


var flake = [];

function bgNight() {
    angle += 0.01;
    var mf = 100; // max flake

    //loop
    for (i = 0; i < mf; i++) {
        flake.push({
            x: Math.random() * w,
            y: Math.random() * h,
            r: Math.floor(Math.random() * 6), //min 2px max 7 px
            d: Math.random() + 1
        })
    }

    function drawflake() {
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "white";
        ctx.beginPath();
        for (var i = 0; i < mf; i++) {

            var f = flake[i];
            ctx.moveTo(f.x, f.y);
            ctx.arc(f.x, f.y, f.r, 0, Math.PI * 2, true);
        }
        ctx.fill();

        moveFlakes();
        ctx.closePath();
    }

    function moveFlakes() {
        for (i = 0; i < mf; i++) {
            var f = flake[i];
            f.y += Math.pow(f.d, 2) + 1;
            f.x += Math.sin(angle) * 2;
            //if snowflake reach to the bottom , send new one to the top
            if (f.y > h) {
                flake[i] = { x: Math.random() * w, y: 0, r: f.r, d: f.d };
            }
        }
    }

    setInterval(drawflake, 25);
}
window.onload = function(){
    bgNight()
}

getEle('theme').addEventListener('change', abc) 

function getEle(id){
    return document.getElementById(id)
}