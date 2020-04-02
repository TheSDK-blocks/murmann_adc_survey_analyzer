module murmann_adc_survey_analyzer( input reset,
                 input A, 
                 output Z );
//reset does nothing
assign Z= !A;

endmodule
