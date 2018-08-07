
var g_instance_id = '';  // <-- Get these values for your IBM Watson
var g_password    = '';  // <-- Machine Learning service credentials
var g_url         = '';  // <-- that are listed in your project README
var g_username    = '';  //


var g_area_action_AI_function_deployment_id = ''; // <-- Get this from the "Business Area and Action" AI function deployment details page


// This sample payload is from section 2 in the "CARS4U-Business Area and Action-AI function" sample notebook
var g_sample_payload = { "fields" : [ "ID", "Gender", "Status", "Children", "Age", "Customer_Status", "Car_Owner", "Customer_Service", "Satisfaction" ],
                         "values" : [ [ 2624, 'Male', 'S', 0, 49.27, 'Active', 'No', "Good experience with all the rental co.'s I contacted. I Just called with rental dates and received pricing and selected rental co.", 1 ] ] 
                       } 
// This example of an invalid payload is missing the customer comment
var g_error_payload =  { "fields" : [ "ID", "Gender", "Status", "Children", "Age", "Customer_Status", "Car_Owner", "Satisfaction" ],
                         "values" : [ [ 2624, 'Male', 'S', 0, 49.27, 'Active', 'No', 1 ] ] 
                       } 

getAuthToken( g_username, g_password ).then( function( token )
{
    //console.log( "\n\nToken:\n\n" + token + "\n\n" );

    processInput( token,
                  g_sample_payload,  // <-- Replace this with g_error_payload to see the error handling
                  g_area_action_AI_function_deployment_id ).then( function( result )
    {
        printResult( result );
        
    } ).catch( function( error )
    {
        printError( "AI function:\n" + error );

    } );

} ).catch( function( token_error )
{
    printError( "Generate token:\n" + token_error );

} );


function getAuthToken( username, password )
{
    // http://watson-ml-aisphere-api-beta.mybluemix.net/#!/Token/generateToken

    return new Promise( function( resolve, reject )
    {
        var btoa = require( 'btoa' );
        var options = { url     : g_url + '/v3/identity/token',
                        headers : { 'Authorization' : 'Basic ' + btoa( username + ":" + password ) } };

        var request = require('request');
        request.get( options, function( error, response, body )
        {
            if( error )
            {
                reject( error );
            }
            else
            {
                resolve( JSON.parse( body ).token );
            }

        } );

    } );    

}


function processInput( token, payload, deployment_id )
{
    // http://watson-ml-aisphere-api-beta.mybluemix.net/#!/Deployments/post_v3_wml_instances_instance_id_deployments_deployment_id_online
    
    return new Promise( function( resolve, reject )
    {
        var options = { url     : g_url + '/v3/wml_instances/' + g_instance_id + '/deployments/' + deployment_id + '/online',
                        headers : { 'Authorization' : 'Bearer ' + token, 'Content-type'  : 'application/json' },
                        body    : JSON.stringify( payload ) };

        var request = require('request');
        request.post( options, function( error, response, body )
        {
            if( error )
            {
                reject( error );
            }
            else
            {
                var result = JSON.parse( body );
                
                if( 'errors' in result )
                {
                    reject( result.errors[0].message );
                }
                else
                {
                    resolve( result );
                }
            }

        } );

    } );    
}


function printError( error_str )
{
    console.log( "\n\nError:\n\n" + error_str );
}


function printResult( result_json )
{
    console.log( "\n\nResult:\n\n" + JSON.stringify( result_json, null, 3 ) );
}

