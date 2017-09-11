#!/usr/bin/env node
var request = require('request');
var fs = require('fs');
var async = require('async');
var url = require('url');
//var URL = 'tegra-ubuntu.local';
var URL = '192.168.1.23';
var PORT = 7090;
var FILENAME = '/tmp/data.calib';
var SHAPE_KINECT = [1280, 720];

var retryWithDelay = function(f, nTimes, delay, cb) {
    async.retry(nTimes,
                function(cb0) {
                    var cb1 = function (err, res) {
                        if (err && (nTimes > 1)) {
                            setTimeout(function() { cb0(err, res); },
                                       delay);
                        } else {
                            cb0(err, res);
                        }
                    };
                    f(cb1);
                }, function(err, res) {
                    cb(err, res);
                });
};

var callPost = function(targetURL, args, cb0) {
/*
    console.log(targetURL);
    console.log(JSON.stringify(args));
    cb0(null);
 */
    var f = function(cb1) {
        try {
            request.post({url: targetURL,
                          json: args},
                         function(err, response, body) {
                             cb1(err, body);
                         });
        } catch (err) {
            cb1(err);
        }
    };
    retryWithDelay(f, 10, 1000, cb0);
};

var cameraCalibratePost = function(objPoints, imagePoints, shape, cb0) {
    var baseURL = 'http://' + URL + ':' + PORT + '/calibrateProjector';
    callPost(baseURL, {objPoints: objPoints, imagePoints: imagePoints,
                       shape: shape}, cb0);
};

var filterNulls = function(x) {
    var nullIndex = {};
    x.points3D.forEach(function(elem, i) {
        if (elem[0] === null) {
            nullIndex[i] = true;
        }
    });

    var result = {points3D: [], points2D: []};

    x.points3D.forEach(function(elem, i) {
        if (!nullIndex[i]) {
            result.points3D.push(elem);
        }
    });

    x.points2D.forEach(function(elem, i) {
        if (!nullIndex[i]) {
            result.points2D.push(elem);
        }
    });

    return result;
};

var extractObjPoints = function(x) {
    return x.map(function(y) {
        console.log(y.points3D.length);
        return y.points3D;
    });
};

var extractImagePoints = function(x) {
    return x.map(function(y) {
        console.log(y.points3D.length);
        return y.points2D;
    });
};


var cameraCalibrate = function(p, cb) {
    var newP = p.map(function(x) { return filterNulls(x);});
    cameraCalibratePost(extractObjPoints(newP), extractImagePoints(newP),
                        SHAPE_KINECT, cb);
};


cameraCalibrate(JSON.parse(fs.readFileSync(FILENAME).toString()),
                function(err, results) {
                    if (err) {
                        console.log(err);
                    } else {
                        console.log(JSON.stringify(results));
                    }
                });
