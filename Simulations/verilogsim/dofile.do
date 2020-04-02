add wave -position insertpoint  \
sim/:tb_murmann_adc_survey_analyzer:A \
sim/:tb_murmann_adc_survey_analyzer:initdone \
sim/:tb_murmann_adc_survey_analyzer:clock \
sim/:tb_murmann_adc_survey_analyzer:Z \

run -all
