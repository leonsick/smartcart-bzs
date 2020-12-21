function register() {
    var cart_code = $('#cart_code').val();
    var current_url = window.location.toString();
    console.log("CURRENT URL: " + current_url)
    var cart_id = current_url.substring(29,30); // EC2
    //var cart_id = current_url.substring(31,32); // LOCAL
    console.log("CART ID: " + cart_id)

    window.location.href = "/register/"+cart_id+"/"+cart_code;
}

// function end() {
//     var current_url = window.location.toString();
//
//     //var cart_id = current_url.substring(30,31); // EC2
//     var cart_id = current_url.substring(32,33); // LOCAL
//
//     //var session_id = current_url.substring(32); // EC2
//     var session_id = current_url.substring(34); // LOCAL
//
//     console.log(cart_id);
//     console.log(session_id);
//
//     window.location.href = "/paypage/"+cart_id+"/"+session_id;
// }
