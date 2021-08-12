


def evaluate_model(Y_teste, X_teste, model):
    
    print('Avaliando modelo:')
    print('====================================================================\n')
    print(classification_report(Y_teste, model.predict(X_teste)))
    print('====================================================================\n')
    ac = metrics.accuracy_score(Y_teste, model.predict(X_teste))
    ac_ = round(ac*100,2)
    print('Obtemos acuárcia de {}%\n'.format(colored(ac_,'blue')))
    print('====================================================================\n')
    mat = confusion_matrix(Y_teste, model.predict(X_teste))
    
    class_names = list(set(Y))
    plot_confusion_matrix(Y_teste, model.predict(X_teste), classes=list(set(Y)), title='Confusion matrix, without normalization')
    plt.show()
    print('\n')
    
    mat = confusion_matrix(Y_teste, model.predict(X_teste))
    print('- {} falsos positivos'.format(mat[0][1])
    print('- {} falsos negativos'.format(mat[1][0])

    print('====================================================================')

