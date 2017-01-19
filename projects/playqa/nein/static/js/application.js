$(function() {
  "use strict";

  var detect = $("#detect");
  var header = $('#header');
  var messages = $('#messages');
  var input = $('#input');
  var inputText = $("#intxt");
  var inputGrid = $('#gdebug');
  var status = $('#status');
  var myName = false;
  var author = null;
  var logged = false;
  var socket = $.atmosphere;
  var subSocket;
//  var transport = 'long-polling';
  var transport = 'websocket';

  var request = {
    url: "/the-chat",
    contentType: "application/json",
    logLevel: 'debug',
    transport: transport,
    fallbackTransport: 'long-polling'
  };

  request.onOpen = function(response) {
    addSysMessage('Atmosphere connected using ' + response.transport);

    detect.hide();
    input.show();

    status.hide();
    transport = response.transport;

    if (response.transport == "local") {
      subSocket.pushLocal("Name?");
    }
  };

  request.onReconnect = function(rq, rs) {
    socket.info("Reconnecting")
  };

  request.onMessage = function(rs) {

    // We need to be logged first.
    if (!myName) return;

    var message = rs.responseBody;
    console.log("onMessage");
    try {
      var json = jQuery.parseJSON(message);
      console.log(json);
    } catch (e) {
      console.log('This doesn\'t look like a valid JSON object: ', message.data);
      return;
    }

    if (!logged) {
      logged = true;
      subSocket.pushLocal(myName);
    } else {
      var me = json.author == author;
      var date = typeof(json.time) == 'string' ? parseInt(json.time) : json.time;
      var world = json.world;
      drawGridFromArray(world.split(", "));
      addMessage(
        {
          type: me ? 'primary' : 'default',
          text: json.author
        },
        json.message,
        new Date(date)
      );
    }
  };

  request.onClose = function(rs) {
    logged = false;
  };

  request.onError = function(rs) {
    messages
      .add('li')
      .addClass('list-group-item')
      .text('Sorry, but there\'s some problem with your socket or the server is down');
  };

  subSocket = socket.subscribe(request);

  inputText.keydown(function(e) {
    if (e.keyCode === 13) {
      var msg = $(this).val();
      // First message is always the author's name
      if (author == null) {
        author = msg;
      }

      var json = {
        author: author,
        message: msg,
        world: unescape(gstr)
      };

      subSocket.push(jQuery.stringifyJSON(json));
      $(this).val('');
      $(this).attr('placeholder',  'Enter your message...');


      if(myName === false) {
        myName = msg;
        logged = true;
        subSocket.pushLocal(myName);
        addSysMessage("You are now known as: " + myName);
      } else {
        addMessage(
          {
            type: "primary",
            text: author
          },
          msg
        );
      }
    }
  });

  function addSysMessage(msg) {
    addMessage(
      {
        type: 'info',
        text: 'System'
      },
      msg
    );
  }

  function addMessage(label, msg, datetime) {
    if(datetime == null) {
        datetime = new Date();
    }
    var time = (datetime.getHours() < 10 ? '0' + datetime.getHours() : datetime.getHours()) + ':' + (datetime.getMinutes() < 10 ? '0' + datetime.getMinutes() : datetime.getMinutes());
    if(label != null) {
      messages
        .append("<li class='list-group-item'><span class='label label-" + label.type + "'>" + label.text + "</span> [" + time + "]: " + msg + "</li>");
    } else {
      messages
        .append("<li class='list-group-item'>[" + time + "]: " + msg + "</li>");
    }
  }
});