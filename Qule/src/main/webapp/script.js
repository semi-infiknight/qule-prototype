$(function () {
  $(".status-button:not(.open)").on("click", function (e) {
    $(".overlay-app").addClass("is-active");
  });
  $(".pop-up .close").click(function () {
    $(".overlay-app").removeClass("is-active");
  });
});

$(".status-button:not(.open)").click(function () {
  $(".pop-up").addClass("visible");
});

$(".pop-up .close").click(function () {
  $(".pop-up").removeClass("visible");
});

let quotesData;
var currentQuote = '',
currentAuthor = '';

function getQuotes() {
  return $.ajax({
    headers: {
      Accept: 'application/json' },

    url:
    'https://gist.githubusercontent.com/semi-infiknight/a7c08bcb64c2a6bcf4978d6add582c60/raw/c25c7c053ee206cb63cfd59436704dbd5d8c0fda/mol.json',
    success: function (jsonQuotes) {
      if (typeof jsonQuotes === 'string') {
        quotesData = JSON.parse(jsonQuotes);
        console.log('quotesData');
        console.log(quotesData);
      }
    } });

}

function getRandomQuote() {
  return quotesData.quotes[
  Math.floor(Math.random() * quotesData.quotes.length)];

}

function getQuote() {
  let randomQuote = getRandomQuote();

  currentQuote = randomQuote.quote;
  currentAuthor = randomQuote.author;
  const image = document.querySelector('[id="im"]');

  $('.quote-text').animate({ opacity: 0 }, 500, function () {
    $(this).animate({ opacity: 1 }, 500);
    image.setAttribute("src", randomQuote.quote);
  });

  $('.quote-author').animate({ opacity: 0 }, 500, function () {
    $(this).animate({ opacity: 1 }, 500);
    $('#author').html(randomQuote.author);
  });

}

$(document).ready(function () {
  getQuotes().then(() => {
    getQuote();
  });

  $('#new-quote').on('click', getQuote);
});

var time_to_complete = 0;
var load_progress = 0;
var loading_interval;

// Cache jQuery:
var loading_text = $('#loading-progress');
var loading_screen = $('.loading-screen');

function beginLoading(time_to_complete) {
  load_progress = 0;

  // Interval to simulate loading.
  loading_interval = setInterval(function () {
    load_progress++;
    loading_text.html(load_progress + '%');

    // Add different stages to the loading screen.
    if (load_progress > 20) loading_screen.addClass('loading-stage-1');
    if (load_progress > 40) loading_screen.addClass('loading-stage-2');
    if (load_progress > 60) loading_screen.addClass('loading-stage-3');
    if (load_progress > 80) loading_screen.addClass('loading-stage-4');

    if (load_progress >= 100) {
      loading_screen.addClass('loading-stage-5');
      finishLoading();
    }

  }, time_to_complete / 100);
}

function finishLoading() {
  loading_text.html('<a style="color:#662d91">Qule</a>');
  clearInterval(loading_interval);
  // Add final stage after short delay.
  setTimeout(function () {
    loading_screen.addClass('loading-stage-6');
  }, 250);
}

// Reset button
$(document).ready(function () {
  $('.reset-button').click(function (event) {
    event.preventDefault();
    loading_screen.removeClass('loading-stage-1 loading-stage-2 loading-stage-3 loading-stage-4 loading-stage-5 loading-stage-6');
    beginLoading(20000);
  });
});

beginLoading(1000);