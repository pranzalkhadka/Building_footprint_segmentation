from keras import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate, Input, Dropout

class Architecture:

    def Conv2D_block(self, input_tensor, n_filters, kernel_size=3, batchnorm=True):
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                kernel_initializer='he_normal', padding='same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                kernel_initializer='he_normal', padding='same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x


    def unet(self, input_image, n_filters=16, kernel_size=3, dropout=0.1, batchnorm=True):
        #Encoder
        c1 = self.Conv2D_block(input_image, n_filters, kernel_size = kernel_size, batchnorm = batchnorm)
        p1 = MaxPooling2D((2,2))(c1)
        p1 = Dropout(dropout)(p1)

        c2 = self.Conv2D_block(p1, n_filters * 2, kernel_size = kernel_size, batchnorm = batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.Conv2D_block(p2, n_filters * 4, kernel_size = kernel_size, batchnorm = batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.Conv2D_block(p3, n_filters * 8, kernel_size = kernel_size, batchnorm = batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = self.Conv2D_block(p4, n_filters * 16, kernel_size = kernel_size, batchnorm = batchnorm)

        # Decoder
        u6 = Conv2DTranspose(n_filters * 8, (kernel_size, kernel_size), strides=(2,2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.Conv2D_block(u6, n_filters * 8, kernel_size = kernel_size, batchnorm = batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (kernel_size, kernel_size), strides=(2,2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.Conv2D_block(u7, n_filters * 4, kernel_size = kernel_size, batchnorm = batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (kernel_size, kernel_size), strides=(2,2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.Conv2D_block(u8, n_filters * 2, kernel_size = kernel_size, batchnorm = batchnorm)

        u9 = Conv2DTranspose(n_filters, (kernel_size, kernel_size), strides=(2,2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.Conv2D_block(u9, n_filters, kernel_size = kernel_size, batchnorm = batchnorm)

        output = Conv2D(1, (1,1), activation='sigmoid')(c9)
        model = Model(inputs=[input_image], outputs=[output])
        return model