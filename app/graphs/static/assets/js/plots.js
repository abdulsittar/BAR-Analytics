
$("#dwnButton").unbind().click(function() {
			topbar.show()
			
			$.ajax({
						url: '/sensoranalysis/downloadEvents',
						type: 'GET',
						contentType: 'application/json;charset=UTF-8',
						data: {
							'selected_event': document.getElementById('phone').value
						},
						dataType: "json",
						success: function (data) {
							console.log("file downloaded complete");
							console.log(data);
							console.log(data.length)
							var x= document.getElementById('customEventSelection');
							var selectStyle = window.getComputedStyle(x);
						    var selectWidth = selectStyle.getPropertyValue("width");

							let keys = Object.keys(data);
							let vals = Object.values(data);
							
							console.log(keys);
							console.log(vals[0]);
							
							for (var i = 0; i<keys.length; i++){
								var opt = document.createElement('option');
								opt.value = keys[i];
								opt.innerHTML = vals[i];
								opt.style.width = "500px";

								
								x.add(opt);
							}
							topbar.hide()
							if(keys.length > 1){
								alert(keys.length + " Events related to the keywords have been downloaded!");
								
							}
							else if(keys.length == 1){
								alert(keys.length + " Event related to the keywords have been downloaded!");
								
							} else {
								alert("Unfortunately, there is no event available for the keywords!");
								
							}
							//$('#hcsg').empty();
							//Plotly.newPlot('hcsg', data)
						},
						error: function (err) {
							topbar.hide()
							alert("Unfortunately, there is no event available for the keywords!");
						}
					});
		});

function BarrierSelectedQA(button){
	$.ajax({
		url: "/sensoranalysis/qaselect",
		type: "GET",
		contentType: 'application/json;charset=UTF-8',
		data: {
			'barrier': document.getElementById('barrier').value
		},
		dataType: "json",
		success: function (data) {
			$('#topicgraph').empty();
			$('#topicdist').empty();
			$('#topictable').empty();
			$('#trendtable').empty();
			$('#linegraph').empty();
		}
	});
};

function getSelectValues(select) {
var result = [];
var options = select.selectedOptions;
var opt;
console.log(options.length)
for (var i=0, iLen=options.length; i<iLen; i++) {
  opt = options[i];
  var a = opt.value
  console.log(a)
  result.push(a);
}
console.log(result.join(","));
return result.join(",");
}

function GetTrendsQA(button){
	topbar.show()
	$.ajax({
		url: '/sensoranalysis/qaline',
		type: 'GET',
		contentType: 'application/json;charset=UTF-8',
		data: {
			  'commaValues': getSelectValues(document.getElementById('commaValues')),
			  'selBarrier': document.getElementById('barrier').value
		},
		dataType: "json",
		success: function (data) {
			var layout = {'xaxis.range': [1, 30]}
			Plotly.newPlot('qalinegraph', data);
			
			$.ajax({
				url: '/sensoranalysis/BertTopicQA',
				type: 'GET',
				contentType: 'application/json;charset=UTF-8',
				data: {
					'commaValues': getSelectValues(document.getElementById('commaValues')),
					'selBarrier': document.getElementById('barrier').value,
					'width' : document.getElementById('qalinegraph').getBoundingClientRect().width,
					'height' : document.getElementById('qalinegraph').getBoundingClientRect().height
				},
				dataType: "json",
				success: function (data) {
					console.log("Button clicked")
					try{
						console.log(data['data'].length)
					}
					catch(e)
					{
						console.log("failed")
					}
					var tot = data.length;
					document.getElementById("qawordcloud").innerHTML = "";
				
				var totalhighet = 0;
				for(x=0; x<tot;x++) {
					var elements = data[x]['data']
					var calh  = (Math.ceil(elements.length/4))*document.getElementById('qalinegraph').getBoundingClientRect().height + 'px';
					totalhighet = totalhighet + calh;
				}
				document.getElementById('qawordcloud').style.height = totalhighet + 'px';
				document.getElementById('qawordcloud').style.width = document.getElementById('qalinegraph').getBoundingClientRect().width + 'px';
				//document.getElementById('qawordcloud').setAttribute('justify-content', 'center');
				//document.getElementById('qawordcloud').setAttribute('align-items', 'center');
				//console.log("total discusions word clouds")
				//console.log(tot)
				
				var x = 0;
				var totalhighet = 0;
				for(x=0; x<tot;x++) {
					var elements = data[x]['data']
					var board = document.createElement('div');
					var idstr = "div_"+x.toString();
					board.id = idstr;
					var calh  = (Math.ceil(elements.length/4))*document.getElementById('qalinegraph').getBoundingClientRect().height + 'px';
					board.style.height = calh
					totalhighet = totalhighet + board.style.height;
					board.style.width = document.getElementById('qawordcloud').style.width;
					document.getElementById('qawordcloud').appendChild(board);
					Plotly.newPlot(idstr, data[x])
					document.getElementById(idstr).setAttribute('align', 'center');
				}
				$("#qawordcloud").children().each(function () {
					$(this).css('text-align','center');
					$(this).css('justify-content','center');
				})
				topbar.hide()
				},
				error: function (err) {
					topbar.hide()
					alert("Unfortunately, failed to create word couds!");
				}
			});
		},
				error: function (err) {
					topbar.hide()
					alert("Unfortunately, failed to create line graph!");
				}
	});
};

$('#trendsQA').on('click', 'button', function () {
	topbar.show()
	$.ajax({
		url: '/sensoranalysis/qaline',
		type: 'GET',
		contentType: 'application/json;charset=UTF-8',
		data: {
			  'commaValues': getSelectValues(document.getElementById('commaValues')),
			  'selBarrier': document.getElementById('barrier').value
		},
		dataType: "json",
		success: function (data) {
			var layout = {'xaxis.range': [1, 30]}
			Plotly.newPlot('qalinegraph', data);
			
			$.ajax({
				url: '/sensoranalysis/BertTopicQA',
				type: 'GET',
				contentType: 'application/json;charset=UTF-8',
				data: {
					'commaValues': getSelectValues(document.getElementById('commaValues')),
					'selBarrier': document.getElementById('barrier').value
				},
				dataType: "json",
				success: function (data) {
					console.log("Button clicked")
					try{
						console.log(data['data'].length)
					}
					catch(e)
					{
						console.log("failed")
					}
					var tot = data.length;
					document.getElementById("qawordcloud").innerHTML = "";
				
				var totalhighet = 0;
				for(x=0; x<tot;x++) {
					var elements = data[x]['data']
					var calh  = (Math.ceil(elements.length/4))*300 + 'px';
					totalhighet = totalhighet + calh;
				}
				document.getElementById('qawordcloud').style.height = totalhighet + 'px';
				document.getElementById('qawordcloud').setAttribute('justify-content', 'center');
				document.getElementById('qawordcloud').setAttribute('align-items', 'center');
				console.log("total discusions word clouds")
				console.log(tot)
				
				var x = 0;
				var totalhighet = 0;
				for(x=0; x<tot;x++) {
					var elements = data[x]['data']
					var board = document.createElement('div');
					var idstr = "div_"+x.toString();
					board.id = idstr;
					var calh  = (Math.ceil(elements.length/4))*300 + 'px';
					board.style.height = calh
					totalhighet = totalhighet + board.style.height;
					board.style.width = document.getElementById('qawordcloud').style.width;
					document.getElementById('qawordcloud').appendChild(board);
					Plotly.newPlot(idstr, data[x])
					document.getElementById(idstr).setAttribute('align', 'center');
				}
				$("#qawordcloud").children().each(function () {
					$(this).css('text-align','center');
					$(this).css('justify-content','center');
				})
				topbar.hide()
				},
				error: function (err) {
					topbar.hide()
					alert("Unfortunately, failed to create word couds!");
				}
			});
		},
				error: function (err) {
					topbar.hide()
					alert("Unfortunately, failed to create line graph!");
				}
	});
});

$('div#ThemeRiverHC').on('plotly_click', function (_, data) {
	topbar.show()
	$.ajax({
		url: '/sensoranalysis/hcThemeRiver',
		type: 'GET',
		contentType: 'application/json;charset=UTF-8',
		data: {
			'x': data.points[0].x
		},
		dataType: "json",
		success: function (data) {
			document.getElementById("label").style.display = 'none';
			Plotly.newPlot('hcsg', data)
			topbar.hide()
		},
				error: function (err) {
					topbar.hide()
					alert("Unfortunately, failed to create hierarchical clustering!");
				}
	});
});