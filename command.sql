delete n1 from training_data_e1 n1, training_data_e1 n2 where n1.id > n2.id and n1.message = n2.message