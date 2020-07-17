# IJS_water_loss

Brailita sample data description

From CUP we have received a data dump of water flow and pressure for the district of Brailita. 
The .json’s file typical row is:


{

"timeStamp":"2018-11-23 09:53:00",

"idflowmeter":"MAG8000_024905H318",

"Tot1":49.18,

"Tot2":0.0,

"Analog2":1.1

}


Parameters:

`“timeStamp”`: is the date time of the mesure.

`“Idflowmeter”`: is the ID of the meter The peculiar series of water measures for two sensor IDs are yet to be explained by Braila.

`“Tot1”`: is the water flow in m3 - Flow in the direction of consumers(households). Can be viewed as flow that goes into the braila district  

`“Tot2”`: is the water flow in m3 - Flow that goes out the braila district.

`“Analog2”`: is the water pressure of the measure point. The unit is probably Pascal, however not yet confirmed by Braila.

Download link for the data:
https://yadi.sk/d/fMKEu9Ahe-TSlg
