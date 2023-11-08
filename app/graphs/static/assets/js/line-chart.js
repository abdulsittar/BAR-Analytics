// Initialize echart object, based on the prepared DOM.
var myChart = echarts.init(document.getElementById('main'));
// var myChart = echarts.init($('#main').get(0));

// Costumize the options and data of the chart.
var option = {
    title: {
        text: 'ECharts 入门示例'
    },
    tooltip: {},
    legend: {
        data:['销量']
    },
    xAxis: {
        data: ["衬衫","羊毛衫","雪纺衫","裤子","高跟鞋","袜子"]
    },
    yAxis: {},
    series: [{
        name: '销量',
        type: 'bar',
        data: [5, 20, 36, 10, 10, 20]
    }]
};

// Render the chart on page, using the former data and options.
myChart.setOption(option);