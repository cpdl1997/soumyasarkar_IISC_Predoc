import torch
import os

def run(_input):
    
    model_year_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../saved_models", "model_year.pt")
    model_year = torch.jit.load(model_year_path)
    model_year.eval() #no more training on this model
    
    model_genre_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../saved_models", "model_genre.pt")
    model_genre = torch.jit.load(model_genre_path)
    model_genre.eval() #no more training on this model

    pred_genre_op = model_genre(_input).tolist()[0]
    pred_year = model_year(_input).tolist()[0][0]

    genre_columns = [ 'Action' ,   'Adventure' ,   'Animation' ,   'Biography' ,   'Comedy' ,   'Crime' ,   'Documentary' ,   'Drama' ,   'Family' ,   'Fantasy' ,   'Film-Noir' ,   'History' ,   'Horror' ,   'Music' ,   'Musical' ,   'Mystery' ,   'Romance' ,   'Sci-Fi' ,   'Sport' ,   'Thriller' ,   'War' ,   'Western' ]
    predicted_genre = []
    for i, val in enumerate(pred_genre_op):
        if val==1:
            predicted_genre.append(genre_columns[i])
    
    return pred_year, predicted_genre

if __name__=="__main__":
    #Take input from user: _input
    _input=0
    pred_year, predicted_genre = run(_input)
    print("Predicted Year = ", pred_year)
    print("Predicted Genre = ", predicted_genre)