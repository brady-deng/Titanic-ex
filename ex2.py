import ex1
item = ['PassengerId','Name','Ticket','Cabin','Embarked']
item_con = ['Sex']
item_mod = ['Age']
item_label = ['Survived']
data_train,label,ind_mod = ex1.load_traindata(item,item_con,item_label,item_mod)
data_train = ex1.modify(data_train,ind_mod,18,60)
# ex1.data_analysis(data_train,2,3)
print(data_train)
test_con = ['Sex']
test_mod = ['Age']
test_del = ['PassengerId','Name','Ticket','Cabin','Embarked']
data_test,ind_mod = ex1.load_testdata(test_del,test_con,test_mod)
data_test = ex1.modify(data_test,ind_mod,18,55)
label_test = ex1.load_testlabel()
data_train = ex1.modify_fare(data_train)
data_test = ex1.modify_fare(data_test)
data_train = ex1.pluse(data_train,0.001)
data_test = ex1.pluse(data_test,0.001)

ex1.train_net(data_train,label,data_test,label_test,50000,0.01)
