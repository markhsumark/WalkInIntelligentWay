$(document).ready(function(){
    $("#btn_start").click(function(){
        const request = new XMLHttpRequest();
        const source = $('#source').val();
        var start_info = {
            'click start': 'true', 
            'source': source
        };
        console.log(start_info);
        request.open('POST', `/StartInfo/${JSON.stringify(start_info)}`)
        request.send()
        
    });

});

