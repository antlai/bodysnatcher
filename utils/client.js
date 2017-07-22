#!/usr/bin/env node
var request = require('request');
var JSONStream = require('jsonstream');
var es = require('event-stream');
var url = require('url');
var URL = 'tegra-ubuntu.local';
var PORT = 7090;

var serviceURL = 'http://' + URL + ':' + PORT +
        '/parts?options={"display": true}';

var r = request({url: serviceURL});

r.pipe(JSONStream.parse())
    .pipe(es.mapSync(function (data) {
        console.log(data);
        return data;
    }));
