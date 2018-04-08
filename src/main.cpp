#include "read_table_data.hpp"

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    std::string imageName(argv[1]);

    ReadTableData readTableData;
    readTableData.execute(imageName);
}


