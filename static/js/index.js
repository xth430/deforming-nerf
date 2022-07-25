// window.HELP_IMPROVE_VIDEOJS = false;

// var INTERP_BASE = "https://homes.cs.washington.edu/~kpar/nerfies/interpolation/stacked";
// var NUM_INTERP_FRAMES = 240;

// var interp_images = [];
// function preloadInterpolationImages() {
//   for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
//     var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
//     interp_images[i] = new Image();
//     interp_images[i].src = path;
//   }
// }

// function setInterpolationImage(i) {
//   var image = interp_images[i];
//   image.ondragstart = function() { return false; };
//   image.oncontextmenu = function() { return false; };
//   $('#interpolation-image-wrapper').empty().append(image);
// }


// $(document).ready(function() {
//     // Check for click events on the navbar burger icon
//     $(".navbar-burger").click(function() {
//       // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
//       $(".navbar-burger").toggleClass("is-active");
//       $(".navbar-menu").toggleClass("is-active");

//     });

//     var options = {
// 			slidesToScroll: 1,
// 			slidesToShow: 3,
// 			loop: true,
// 			infinite: true,
// 			autoplay: false,
// 			autoplaySpeed: 3000,
//     }

// 		// Initialize all div with carousel class
//     var carousels = bulmaCarousel.attach('.carousel', options);

//     // Loop on each carousel initialized
//     for(var i = 0; i < carousels.length; i++) {
//     	// Add listener to  event
//     	carousels[i].on('before:show', state => {
//     		console.log(state);
//     	});
//     }

//     // Access to bulmaCarousel instance of an element
//     var element = document.querySelector('#my-element');
//     if (element && element.bulmaCarousel) {
//     	// bulmaCarousel instance is available as element.bulmaCarousel
//     	element.bulmaCarousel.on('before-show', function(state) {
//     		console.log(state);
//     	});
//     }

//     /*var player = document.getElementById('interpolation-video');
//     player.addEventListener('loadedmetadata', function() {
//       $('#interpolation-slider').on('input', function(event) {
//         console.log(this.value, player.duration);
//         player.currentTime = player.duration / 100 * this.value;
//       })
//     }, false);*/
//     preloadInterpolationImages();

//     $('#interpolation-slider').on('input', function(event) {
//       setInterpolationImage(this.value);
//     });
//     setInterpolationImage(0);
//     $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

//     bulmaSlider.attach();

// })


window.HELP_IMPROVE_VIDEOJS = false;

// var SHAPE_INTERP_BASE = "static/data/shape_interpolation";
// var POSE_INTERP_BASE = "static/data/pose_interpolation";
// var NUM_SHAPE_INTERP_FRAMES_SHAE_TO_REGINA = 61;
// var NUM_SHAPE_INTERP_FRAMES_SUZIE_TO_RACER = 61;
// var NUM_POSE_INTERP_FRAMES_ADAM = 121;
// var NUM_POSE_INTERP_FRAMES_SWATGUY = 121;

// var interp_shape_images_shae_to_regina = [];
// var interp_shape_images_suzie_to_racer = [];
// var interp_pose_images_adam = [];
// var interp_pose_images_swatguy = [];

var INTERP_BASE = "static/images/interpolate";
var NUM_INTERP = 60;
var interp_nerf_lego = [];
var interp_nerf_chair = [];

function preloadInterpolationImages() {
    // nerf
    for (var i = 0; i < NUM_INTERP; i++) {
        var path = INTERP_BASE + '/nerf_lego/' + String(i).padStart(4, '0') + '.png';
        interp_nerf_lego[i] = new Image();
        interp_nerf_lego[i].src = path;
    }

    for (var i = 0; i < NUM_INTERP; i++) {
        var path = INTERP_BASE + '/nerf_chair/' + String(i).padStart(4, '0') + '.png';

        interp_nerf_chair[i] = new Image();
        interp_nerf_chair[i].src = path;
    }

    // // Pose
    // for (var i = 0; i < NUM_POSE_INTERP_FRAMES_ADAM; i++) {
    //     var path = POSE_INTERP_BASE + '/adam/interp_' + String(i) + '.jpg';
    //     interp_pose_images_adam[i] = new Image();
    //     interp_pose_images_adam[i].src = path;
    // }

    // for (var i = 0; i < NUM_POSE_INTERP_FRAMES_SWATGUY; i++) {
    //     var path = POSE_INTERP_BASE + '/swatguy/interp_' + String(i) + '.jpg';
    //     interp_pose_images_swatguy[i] = new Image();
    //     interp_pose_images_swatguy[i].src = path;
    // }
}



// SHAPE - shape to regina
function setInterpolationNeRFLego(i) {
    var image = interp_nerf_lego[i];
    image.ondragstart = function() { return false; };
    image.oncontextmenu = function() { return false; };
    $('#interpolation-image-wrapper-nerf-lego').empty().append(image);
}

function setInterpolationNeRFChair(i) {
    var image = interp_nerf_chair[i];
    image.ondragstart = function() { return false; };
    image.oncontextmenu = function() { return false; };
    $('#interpolation-image-wrapper-nerf-chair').empty().append(image);
}

// // SHAPE - shape to regina
// function setInterpolationImageShapeShae2Regina(i) {
//     var image = interp_shape_images_shae_to_regina[i];
//     image.ondragstart = function() { return false; };
//     image.oncontextmenu = function() { return false; };
//     $('#interpolation-image-wrapper-shape-shae-to-regina').empty().append(image);
// }

// function setInterpolationImageShapeSuzie2Racer(i) {
//     var image = interp_shape_images_suzie_to_racer[i];
//     image.ondragstart = function() { return false; };
//     image.oncontextmenu = function() { return false; };
//     $('#interpolation-image-wrapper-shape-suzie-to-racer').empty().append(image);
// }

// // POSE - adam
// function setInterpolationImagePoseAdam(i) {
//     var image = interp_pose_images_adam[i];
//     image.ondragstart = function() { return false; };
//     image.oncontextmenu = function() { return false; };
//     $('#interpolation-image-wrapper-pose-adam').empty().append(image);
// }

// // POSE - swatguy
// function setInterpolationImagePoseSwatguy(i) {
//     var image = interp_pose_images_swatguy[i];
//     image.ondragstart = function() { return false; };
//     image.oncontextmenu = function() { return false; };
//     $('#interpolation-image-wrapper-pose-swatguy').empty().append(image);
// }


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        $(".navbar-burger").toggleClass("is-active");
        $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 3000,
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for (var i = 0; i < carousels.length; i++) {
        // Add listener to  event
        carousels[i].on('before:show', state => {
            console.log(state);
        });
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
        // bulmaCarousel instance is available as element.bulmaCarousel
        element.bulmaCarousel.on('before-show', function(state) {
            console.log(state);
        });
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    // SHAPE - shape to regina
    $('#interpolation-slider-nerf-lego').on('input', function(event) {
        setInterpolationNeRFLego(this.value);
    });
    setInterpolationNeRFLego(0);
    $('#interpolation-slider-nerf-lego').prop('max', NUM_INTERP - 1);

    // SHAPE - suzie to racer
    $('#interpolation-slider-nerf-chair').on('input', function(event) {
        setInterpolationNeRFChair(this.value);
    });
    setInterpolationNeRFChair(0);
    $('#interpolation-slider-nerf-chair').prop('max', NUM_INTERP - 1);


    // // SHAPE - shape to regina
    // $('#interpolation-slider-shape-shae-to-regina').on('input', function(event) {
    //     setInterpolationImageShapeShae2Regina(this.value);
    // });
    // setInterpolationImageShapeShae2Regina(0);
    // $('#interpolation-slider-shape-shae-to-regina').prop('max', NUM_SHAPE_INTERP_FRAMES_SHAE_TO_REGINA - 1);

    // // SHAPE - suzie to racer
    // $('#interpolation-slider-shape-suzie-to-racer').on('input', function(event) {
    //     setInterpolationImageShapeSuzie2Racer(this.value);
    // });
    // setInterpolationImageShapeSuzie2Racer(0);
    // $('#interpolation-slider-shape-suzie-to-racer').prop('max', NUM_SHAPE_INTERP_FRAMES_SUZIE_TO_RACER - 1);


    // // POSE - adam
    // $('#interpolation-slider-pose-adam').on('input', function(event) {
    //     setInterpolationImagePoseAdam(this.value);
    // });
    // setInterpolationImagePoseAdam(0);
    // $('#interpolation-slider-pose-adam').prop('max', NUM_POSE_INTERP_FRAMES_ADAM - 1);

    // // POSE - swatguy
    // $('#interpolation-slider-pose-swatguy').on('input', function(event) {
    //     setInterpolationImagePoseSwatguy(this.value);
    // });
    // setInterpolationImagePoseSwatguy(0);
    // $('#interpolation-slider-pose-swatguy').prop('max', NUM_POSE_INTERP_FRAMES_SWATGUY - 1);


    bulmaSlider.attach();

})