//Bennygmate

#include "Functions.h"
#include "Traditional_NN_layer.h"
#include "Convolution.h"

namespace neurons {
    class CNN_layer : public Traditional_NN_layer    {
    private:
        Conv_2d m_conv2d;

    public:
        static std::string from_binary_data(
            char * binary_data, lint & data_size, TMatrix<> & w, TMatrix<> & b,
            lint & stride, lint & padding,
            std::unique_ptr<Activation> & act_func, std::unique_ptr<ErrorFunction> & err_func,
            char *& residual_data, lint & residual_len
        );

        CNN_layer();

        CNN_layer(
            double mmt_rate,
            lint rows,  
            lint cols, 
            lint chls,  
            lint filters,  
            lint filter_rows, 
            lint filter_cols, 
            lint stride,  
            lint padding, 
            lint threads,  
            neurons::Activation *act_func,  
            neurons::ErrorFunction *err_func = nullptr 
        );

        CNN_layer(double mmt_rate,
            lint rows, lint cols, lint chls, lint stride, lint padding, lint threads,
            const TMatrix<> & w, const TMatrix<> & b,
            std::unique_ptr<Activation> & act_func, std::unique_ptr<ErrorFunction> & err_func);

        CNN_layer(const CNN_layer & other);

        CNN_layer(CNN_layer && other);

        CNN_layer & operator = (const CNN_layer & other);

        CNN_layer & operator = (CNN_layer && other);

        Shape output_shape() const;

        virtual std::string nn_type() const { return NN_layer::CNN; }

        virtual std::unique_ptr<char[]> to_binary_data(lint & data_size) const;
    };

    class CNN_layer_op : public Traditional_NN_layer_op    {
    private:
        std::vector<TMatrix<>> m_conv_to_x_diffs;
        std::vector<TMatrix<>> m_conv_to_w_diffs;
        std::vector<TMatrix<>> m_act_diffs;

        Conv_2d m_conv2d;

    public:
        CNN_layer_op();
        CNN_layer_op(
            const Conv_2d & conv2d,
            const TMatrix<> &w,
            const TMatrix<> &b,
            const std::unique_ptr<Activation> &act_func,
            const std::unique_ptr<ErrorFunction> &err_func);

        CNN_layer_op(const CNN_layer_op & other);

        CNN_layer_op(CNN_layer_op && other);

        CNN_layer_op & operator = (const CNN_layer_op & other);

        CNN_layer_op & operator = (CNN_layer_op && other);

        // Forward prop
        virtual std::vector<TMatrix<>> batch_forward_propagate(const std::vector<TMatrix<>> & inputs);
        virtual std::vector<TMatrix<>> batch_forward_propagate(const std::vector<TMatrix<>> & inputs, const std::vector<TMatrix<>> & targets);

        // Back prop
        virtual std::vector<TMatrix<>> batch_back_propagate(double l_rate, const std::vector<TMatrix<>> & E_to_y_diffs);
        virtual std::vector<TMatrix<>> batch_back_propagate(double l_rate);

        virtual Shape output_shape() const;
    };
}

