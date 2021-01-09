//
// Created by bartosz on 1/1/21.
//

#ifndef TRIANGLE_RAYCASTING_COLOR_CUH
#define TRIANGLE_RAYCASTING_COLOR_CUH


class Color
{
public:
	unsigned char red;
	unsigned char green;
	unsigned char blue;

	__host__ __device__
	inline Color( unsigned char red, unsigned char green, unsigned char blue )
			: red( red ), green( green ), blue( blue )
	{}

	__host__ __device__
	inline Color() : Color( 0, 0, 0 )
	{}

	__host__ __device__
	inline Color operator+( const Color & other ) const
	{
		return Color( red + other.red, green + other.green, blue + other.blue );
	}

	__host__ __device__
	inline Color operator*( const Color & other ) const
	{
		return Color( red * other.red, green * other.green, blue * other.blue );
	}

	__host__ __device__
	inline Color operator/( float a ) const
	{
		return Color( red / a, green / a, blue / a );
	}


	__host__ __device__
	inline Color & operator+=( const Color & other )
	{
		this->red += other.red;
		this->green += other.green;
		this->blue += other.blue;
		return *this;
	}

	__host__ __device__
	inline Color & operator*=( float a )
	{
		this->red *= a;
		this->green *= a;
		this->blue *= a;
		return *this;
	}

	__host__ __device__
	inline Color & operator/=( float a )
	{
		this->red /= a;
		this->green /= a;
		this->blue /= a;
		return *this;
	}
};


__host__ __device__
inline Color operator*( float a, const Color & color )
{
	return Color( a * color.red, a * color.green, a * color.blue );
}


#endif //TRIANGLE_RAYCASTING_COLOR_CUH
