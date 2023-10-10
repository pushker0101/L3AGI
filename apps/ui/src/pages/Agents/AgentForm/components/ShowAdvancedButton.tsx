import styled from 'styled-components'

import NavigationChevronUp from '@l3-lib/ui-core/dist/icons/NavigationChevronUp'
import NavigationChevronDown from '@l3-lib/ui-core/dist/icons/NavigationChevronDown'

import Typography from '@l3-lib/ui-core/dist/Typography'
import TypographyPrimary from 'components/Typography/Primary'

const ShowAdvancedButton = ({ onClick, isShow }: { onClick: () => void; isShow: boolean }) => {
  return (
    <StyledAdvancedButton onClick={onClick}>
      <TypographyPrimary
        value='Advanced Options'
        type={Typography.types.LABEL}
        size={Typography.sizes.md}
      />
      {isShow ? <StyledNavigationChevronDown /> : <StyledNavigationChevronUp />}
    </StyledAdvancedButton>
  )
}
export default ShowAdvancedButton

const StyledAdvancedButton = styled.div`
  cursor: pointer;
  display: flex;
  align-items: center;

  gap: 5px;
`

const StyledNavigationChevronUp = styled(NavigationChevronUp)`
  path {
    color: ${({ theme }) => theme.body.iconColor};
  }
`
const StyledNavigationChevronDown = styled(NavigationChevronDown)`
  path {
    color: ${({ theme }) => theme.body.iconColor};
  }
`